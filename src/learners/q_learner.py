import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class QLearner:
    def __init__(self, mac, scheme, logger, args, mode="camera"):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        if mode == "camera":
            self.n_agent = args.n_agents
            self.name = {"prefix": "", "reward": "reward", "actions": "actions", "avail_actions": "avail_actions",
                         "save_mixer": "mixer.th", "save_opt": "opt.th"}
        elif mode == "target":
            self.n_agent = args.n_target_agents
            self.name = {"prefix": "target_", "reward": "target_reward", "actions": "target_actions",
                         "avail_actions": "target_avail_actions", "save_mixer": "target_mixer.th",
                         "save_opt": "target_opt.th"}
        else:
            raise ValueError("mode should in [camera, target] .")
        
        
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch[self.name["reward"]][:, :-1]
        actions = batch[self.name["actions"]][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch[self.name["avail_actions"]]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            raise NotImplementedError
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        # targets = rewards.squeeze(3) + self.args.gamma * (1 - terminated) * target_max_qvals
        targets = rewards.squeeze(3)[:, :-3, :] + self.args.gamma * rewards.squeeze(3)[:, 1:-2, :] + \
                  self.args.gamma * self.args.gamma * rewards.squeeze(3)[:, 2:-1, :] + \
                  self.args.gamma * self.args.gamma * self.args.gamma * rewards.squeeze(3)[:, 3:, :] + \
                  self.args.gamma * self.args.gamma * self.args.gamma * self.args.gamma * (1 - terminated[:, 3:, :]) * target_max_qvals[:, 3:, :]
        chosen_action_qvals = chosen_action_qvals[:, 3:, :]
        mask = mask[:, 3:, :]

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat(self.name["prefix"] + "loss", loss.item(), t_env)
            self.logger.log_stat(self.name["prefix"] + "grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(self.name["prefix"] + "td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat(self.name["prefix"] + "q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.n_agent), t_env)
            self.logger.log_stat(self.name["prefix"] + "target_mean", (targets * mask).sum().item()/(mask_elems * self.n_agent), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/{}".format(path, self.name["save_mixer"]))
        th.save(self.optimiser.state_dict(), "{}/{}".format(path, self.name["save_opt"]))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/{}".format(path, self.name["save_mixer"]), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/{}".format(path, self.name["save_opt"]), map_location=lambda storage, loc: storage))
