from pettingzoo.utils import wrappers

from typing import Dict, Any, Tuple
from pettingzoo.utils.env import AgentID


class GymStyleWrapper(wrappers.BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
    
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
        info = dict()


        return observations, rewards, dones, info