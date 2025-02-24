# from envs import REGISTRY as env_REGISTRY
from ENV.environment import create_env
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th
import time


def team_spirit(rew, team_rew, tau):
    rew = np.array(rew)
    return (1 - tau) * rew + tau * team_rew


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.env = create_env(self.args.env_id, args=self.args)
        self.env_info = self.env.get_env_info()
        self.episode_limit = self.env_info['episode_limit']
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        # 把一个函数的某些参数给固定住，返回一个新的函数
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac[0]
        self.t_mac = mac[1]

    def get_env_info(self):
        return self.env.get_env_info()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        self.t_mac.init_hidden(batch_size=self.batch_size)
        time_count = 0

        while not terminated:

            pre_transition_data = {
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_centralized_obs()],
                "target_avail_actions": [self.env.get_target_avail_actions()],
                "target_obs": [self.env.get_target_obs()]
            }
            
            self.batch.update(pre_transition_data, ts=self.t)

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            t_actions = self.t_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            while time.time() - time_count < self.args.eps_time:
                time.sleep(0.1)
            # if self.t % 20 == 0:
            #     print("STEP: {}, USE_TIME: {}".format(self.t, time.time() - time_count))
            time_count = time.time()
            _, _, terminated, info = self.env.step([actions[0], t_actions[0]])
            reward = info['Reward']
            team_reward = info['Team_Reward']
            
            episode_return += team_reward

            post_transition_data = {
                "actions": actions,
                "reward": [team_spirit(reward, team_reward, self.args.tau)],
                "target_actions": t_actions,
                "target_reward": [info['Target_Reward']],
                "terminated": [(terminated,)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_centralized_obs()],
            "target_avail_actions": [self.env.get_target_avail_actions()],
            "target_obs": [self.env.get_target_obs()]
        }

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)
        t_actions = self.t_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"target_actions": t_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        cur_stats["ep_rew_mean"] = episode_return / self.t + cur_stats.get("ep_rew_mean", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            if hasattr(self.t_mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.t_mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v/stats["n_episodes"], self.t_env)
        stats.clear()

