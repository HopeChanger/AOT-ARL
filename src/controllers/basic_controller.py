from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args, mode="camera"):
        self.args = args
        if mode == "camera":
            self.n_agents = args.n_agents
            self.name = {"avail_actions": "avail_actions", "save_name": "agent.th", "net_type": "cnn"}
            input_channel = scheme["obs"]["vshape"][0]
            self.agent = agent_REGISTRY["cnn"](input_channel, self.args)
        elif mode == "target":
            self.n_agents = args.n_target_agents
            self.name = {"avail_actions": "target_avail_actions", "save_name": "target_agent.th", "net_type": "rnn"}
            input_channel = scheme["target_obs"]["vshape"][0]
            self.agent = agent_REGISTRY["rnn"](input_channel, self.args)
        else:
            raise ValueError("mode should in [camera, target] .")

        print("input channel: ", input_channel)

        self.agent_output_type = args.agent_output_type  # q
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch[self.name["avail_actions"]][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch[self.name["avail_actions"]][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        # if self.agent_output_type == "pi_logits":
        #
        #     if getattr(self.args, "mask_before_softmax", True):
        #         # Make the logits for unavailable actions very negative to minimise their affect on the softmax
        #         reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
        #         agent_outs[reshaped_avail_actions == 0] = -1e10
        #
        #     agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        #     if not test_mode:
        #         # Epsilon floor
        #         epsilon_action_num = agent_outs.size(-1)
        #         if getattr(self.args, "mask_before_softmax", True):
        #             # With probability epsilon, we will pick an available action uniformly
        #             epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()
        #
        #         agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
        #                        + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)
        #
        #         if getattr(self.args, "mask_before_softmax", True):
        #             # Zero out the unavailable actions
        #             agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def read_hidden(self):
        return self.hidden_states

    def load_hidden(self, hidden):
        self.hidden_states = hidden

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/{}".format(path, self.name["save_name"]))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/{}".format(path, self.name["save_name"]), map_location=lambda storage, loc: storage))

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []

        if self.name["net_type"] == "cnn":
            inputs.append(batch["obs"][:, t])
            inputs = th.cat([x.reshape(bs * self.n_agents, -1, self.args.input_size,  self.args.input_size) for x in inputs], dim=1)
        else:
            inputs.append(batch["target_obs"][:, t])
            inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

        return inputs
