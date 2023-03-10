import numpy as np

import gym
from gym.spaces import Box

from MPECoinGames.game_fixed_coins.scenario import raw_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.utils.wrappers import ClipOutOfBoundsWrapper, AssertOutOfBoundsWrapper,OrderEnforcingWrapper
from pettingzoo.utils import wrappers

from typing import Dict, Any, Tuple

#* it is not necessary for ray training, this func is used only for earlier experiment
def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = ClipOutOfBoundsWrapper(env)
        else:
            env = AssertOutOfBoundsWrapper(env)
        env = OrderEnforcingWrapper(env)
        env = GymStyleWrapper(env)
        return env

    return env

env_fn = make_env(raw_env)
'''
generates one coin once, when the coin is picked by an agent, current episode ends. 
'''
parallel_env_fn = parallel_wrapper_fn(env_fn)


from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from pettingzoo.mpe._mpe_utils.core import World
from MPECoinGames.game_fixed_coins.scenario import Scenario
class GymStyleWrapper(wrappers.BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        _env = env.unwrapped
        while _env is not env.unwrapped:
            _env = _env.unwrapped
        
        self._world = _env.world
        self._scenario = _env.scenario
        
        self.agents = _env.agents

        self._last_info = dict()

    def reset(self):
        self.env.reset()
        observations = {agent: self.env.observe(agent) for agent in self.env.agents}

        self._last_info = {agent: [0, 0] for agent in self.env.agents}

        return observations

    def step(self, actions)-> Tuple[Dict[str, Any], ...]:
        '''
        a gym style API, 'actions' is dictionary of action of each agent,
        and its returns are:
            `observations`  dictionary of observation of each agent.\n
            `rewards`   dictionary of reward of each agent.\n
            `dones`     dictionary of done of each agent.\n
            `info`      dictionary of eatra informations, including state, and which of agents gets the coin.\n
        '''
        for _ in range(self.num_agents):
            agent = self.env.agent_selection
            self.env.step(actions[agent])

        observations = {agent: self.env.observe(agent) for agent in self.env.agents}
        rewards = {agent: self.env.rewards[agent] for agent in self.env.agents}

        # done = -0.5 < observations[agent][4+2+2] and observations[agent][4+2+2] < 0.5   
        # wether the type of the nearest coin is `0` in the observation of (arbitrary) agent
        # if true, then all the coins were picked up (details see scenario)

        # dones = {agent: done for agent in self.agents}
        dones = {agent: self.env.dones[agent] for agent in self.agents}
        info = self._scenario.info
        info['pick'] = False
        for agent in self.agents:
            for _p, p in zip(self._last_info[agent], info[agent]):
                if _p != p:
                    info['pick'] = True
                    break

        self._last_info = info.copy()

        return observations, rewards, dones, info
