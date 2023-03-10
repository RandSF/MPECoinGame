import numpy as np

import gym
from gym.spaces import Box

from MPECoinGames.game_infinite_coins.scenario import raw_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.utils.wrappers import ClipOutOfBoundsWrapper, AssertOutOfBoundsWrapper,OrderEnforcingWrapper
from pettingzoo.utils import wrappers

from typing import Dict, Any, Tuple
from pettingzoo.utils.env import AgentID



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



class GymStyleWrapper(wrappers.BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        _env = env.unwrapped
        while _env is not env.unwrapped:
            _env = _env.unwrapped
        
        self._world = _env.world
        self._scenario = _env.scenario

    
    def reset(self):
        self.env.reset()
        observations = {agent: self.env.observe(agent) for agent in self.agents}
        # info = {'state': self.state()}

        return observations

    def step(self, actions)-> Tuple[Dict[AgentID, Any], ...]:
        '''
        a gym style API, 'actions' is dictionary of action of each agent,
        and its returns are:
            `observations`  dictionary of observation of each agent.\n
            `rewards`   dictionary of reward of each agent.\n
            `dones`     dictionary of done of each agent.\n
            `info`      dictionary of eatra informations, including state, and which of agents gets the coin.\n
        '''
        for _ in range(self.num_agents):
            agent = self.agent_selection
            self.env.step(actions[agent])

        observations = {agent: self.env.observe(agent) for agent in self.agents}
        rewards = {agent: self.env.rewards[agent] for agent in self.agents}

        # done = False
        # for v in rewards.values():
        #     if v != 0:
        #         done = True 
        #         break

        # dones = {agent: done for agent in self.agents}
        dones = {agent: self.env.terminations[agent] for agent in self.agents}
        info = self._scenario.info
        info['pick'] = False
        for agent in self.agents:
            if len(info[agent]) > 0:
                info['pick'] = True
                break


        return observations, rewards, dones, info