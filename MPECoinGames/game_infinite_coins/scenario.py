from typing import List

import numpy as np

from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env


class raw_env(SimpleEnv):
    def __init__(self, max_cycles=500, continuous_actions=True, render_mode=None):
        scenario = Scenario()
        world = scenario.make_world()
        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "coin_game"


env = make_env(raw_env)

parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_agents_per_type = 1):
        world = World()
        # basic flags
        self.has_coin = False
        self.types = ['red', 'blue']
        self.check_coin_count = 0
        self.num_agents = num_agents_per_type*2
        # add agents
        world.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.type = 0 if i < num_agents_per_type else 1
            index = num_agents_per_type - i if agent.type else i
            agent.name = f"{self.types[agent.type]}_agent_{index}"
            agent.collide = True
            agent.silent = True
        #* coin will be added in `.reset_world()`
        world.landmarks = [Landmark()]
        self.info: dict[str, list[bool]] = dict()
        # self.info[agent] = [True] OR [False] OR []
        return world

    def reset_world(self, world, np_random):
        self.check_coin_count = 0
        self.np_random = np_random
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = [np.array([1, 0, 0]), np.array([0, 0, 1])][agent.type]
        # random properties for landmarks
        for landmark in world.landmarks:
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.size = 0.03    # I could not find how to control the `alpha` of entities
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        
        self.generate_coin(world)
        self.info: dict[str, list[bool]] = {agent.name: [] for agent in world.agents}
        self.checked_rew = {agent: False for agent in world.agents}
        self.cur_rew = {agent: 0 for agent in world.agents}

    def reward(self, agent: Agent, world: World):
        _agent = agent
        if self.checked_rew[agent]:
             # if every agent called the `.reward()` function, recaculate the latest collision state of coins
            self.info: dict[str, list[bool]] = {agent.name: [] for agent in world.agents}
            self.checked_rew = {agent: False for agent in world.agents}
            self.cur_rew = {agent: 0 for agent in world.agents}

            coin = world.landmarks[0]   # there is only one landmark
            flag = False # wether the coin is picked
            for agent in world.agents:
                dis_min = agent.size + coin.size
                dis = np.linalg.norm(agent.state.p_pos - coin.state.p_pos)
                if dis < dis_min:
                    flag = True
                    self.info[agent.name].append(coin.type == agent.type)
                    self.cur_rew[agent] += 1
                    # once pick up a coin, get 1 reward
                    if coin.type != agent.type:
                        # if pick up others' coin, others get -1 punishment
                        for other in world.agents:
                            if other.type != agent.type:
                                self.cur_rew[other] -= 1
            if flag:
                self.generate_coin(world)

        self.checked_rew[_agent] = True
        reward = self.cur_rew[_agent]
        return reward
        # reward = 0
        # coin = world.landmarks[0]
        # flags = self.if_gets_coin(coin, world.agents)
        # opponent = None
        # for temp in world.agents:
        #     if temp is not agent:
        #         opponent = temp
        #         break
        # if flags[agent]:
        #     reward += 1
        #     self.check_coin_count += 1
        # if flags[opponent] and (agent.type == coin.type):
        #     reward -= 1
        #     self.check_coin_count += 1
        
        # if self.check_coin_count == self.num_agents:
        #     self.generate_coin(world)
        
        # return reward

    def observation(self, agent: Agent, world: World):
        # get positions of all entities in this agent's reference frame
        self_pos = []
        self_vel = []
        oppo_pos = []
        coin_pos = []
        is_my_coin = [] # if the coin has the same type as agent's
        for entity in world.agents:
            if entity is agent:
                self_pos.append(entity.state.p_pos)
                self_vel.append(entity.state.p_vel)
            else:
                oppo_pos.append(entity.state.p_pos - agent.state.p_pos)
        for entity in world.landmarks:
            coin_pos.append(entity.state.p_pos - agent.state.p_pos)
            is_my_coin.append(np.ones(1) if entity.type == agent.type else np.zeros(1))
        return np.concatenate(self_pos + self_vel + oppo_pos + coin_pos + is_my_coin)

    def generate_coin(self,world):
        self.check_coin_count = 0
        np_random: np.random.Generator = self.np_random

        for coin in world.landmarks:
            while self.check_collide(coin, world.agents):
                coin.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            coin.type = np_random.integers(0, 2)
            coin.color = np.array([1, 0, 0])if coin.type == 0 else np.array([0, 0, 1])

    def check_collide(self, coin: Landmark, agents: List[Agent]):
        for agent in agents:
            dis = np.sum(np.square(agent.state.p_pos - coin.state.p_pos))
            dis_min = np.square(agent.size + coin.size)
            if dis < dis_min:
                return True
        return False

    def if_gets_coin(self, coin: Landmark, agents: List[Agent]):
        # returns results if each agent gets coin
        flags = dict()
        for agent in agents:
            dis = np.sum(np.square(agent.state.p_pos - coin.state.p_pos))
            dis_min = np.square(agent.size + coin.size)
            flags[agent] = True if dis < dis_min else False
        return flags
            

