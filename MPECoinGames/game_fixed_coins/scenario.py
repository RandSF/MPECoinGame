from typing import List
import copy

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World, Entity
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env


class raw_env(SimpleEnv):
    def __init__(self, max_cycles=500, num_coins = 20, continuous_actions=True, render_mode=None):
        EzPickle.__init__(self, max_cycles, num_coins, continuous_actions, render_mode)
        scenario = Scenario()
        world = scenario.make_world(num_coins = num_coins)
        super().__init__(
            scenario=scenario,
            world=world,
            # render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.infos = scenario.infos
        self.metadata["name"] = "coin_game"

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.infos = copy.deepcopy(self.scenario.infos)
            self.steps += 1
            if self.steps >= self.max_cycles or (len(self.world.landmarks) == 0):  # if all coin were picked, the game terminals
                for a in self.agents:
                    self.dones[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_agents_per_type = 1, num_coins = 20,
                        observe_range = 0.5, num_goals = 5):
        world = World()
        # basic flags
        self.has_coin = False
        self.types = ['red', 'blue']
        # 
        self.num_agents = num_agents_per_type*2
        self.num_coins = num_coins
        self.observe_range = observe_range
        self.num_goals = num_goals
        self.init_bound = max(1, num_coins / 20)
        # add agents
        world.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.type = 0 if i < num_agents_per_type else 1
            index = num_agents_per_type - i if agent.type else i
            agent.name = f"{self.types[agent.type]}_agent_{index}"
            agent.collide = True
            agent.silent = True
        #* coin will be added in `.reset_world()`

        # used for send information to Env during each `.step()`
        self.info: dict[str, list[int]] = dict()  
        self.infos = {}
        # self.info[agent] = [num1, num2]
        # num1 and num2 are int, which means, correspondly, the total number of picked coins with the same color and the number of those with different color 
        return world

    def reset_world(self, world, np_random):
        self.np_random = np_random
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = [np.array([1, 0, 0]), np.array([0, 0, 1])][agent.type]
        # random properties for landmarks
        world.landmarks = [Landmark() for _ in range(self.num_coins)]
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-self.init_bound, +self.init_bound, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.type = None    # to be set in .generate_coin()
            landmark.type = 0 if i < self.num_coins/2 else 1  #* equal number of both type
            landmark.name = f"{self.types[landmark.type]}_coin"
            landmark.size = 0.03    # I could not find how to control the `alpha` of entities
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-self.init_bound, +self.init_bound, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        
        self.generate_coin(world)
        self.info: dict[str, list[int]] = {agent.name: [0, 0] for agent in world.agents}
        self.infos = {agent.name: self.info for agent in world.agents}    # lazy reference
        self.checked_rew = {agent: False for agent in world.agents}
        self.cur_rew = {agent: 0 for agent in world.agents}

    def reward(self, agent: Agent, world: World):
        _agent = agent
        if self.checked_rew[agent]:
            # if every agent called the `.reward()` function, recaculate the latest collision state of coins
            # self.info: dict[str, list[bool]] = {agent.name: [] for agent in world.agents}
            self.checked_rew = {agent: False for agent in world.agents}
            self.cur_rew = {agent: 0 for agent in world.agents}

            left_coins = []
            for coin in world.landmarks:
                flag = False
                for agent in world.agents:
                    dis_min = agent.size + coin.size
                    dis = np.linalg.norm(agent.state.p_pos - coin.state.p_pos)
                    if dis < dis_min:
                        flag = True
                        self.info[agent.name][coin.type == agent.type] += 1
                        self.cur_rew[agent] += 1
                        # once pick up a coin, get 1 reward
                        if coin.type != agent.type:
                            # if pick up others' coin, others get -1 punishment
                            for other in world.agents:
                                if other.type != agent.type:
                                    self.cur_rew[other] -= 1
                if not flag:
                    left_coins.append(coin)

            # delete picked-up coin
            world.landmarks = left_coins

        self.checked_rew[_agent] = True
        reward = self.cur_rew[_agent]
        return reward

    def observation(self, agent: Agent, world: World):
        # get positions of all entities in this agent's reference frame
        self_pos = []
        self_vel = []
        oppo_pos = []
        coin_obs = []
        num_goal = self.num_goals
        for entity in world.agents:
            if entity is agent:
                self_pos.append(entity.state.p_pos)
                self_vel.append(entity.state.p_vel)
            else:
                oppo_pos.append(entity.state.p_pos - agent.state.p_pos)

        dis = []
        for coin in world.landmarks:
            same_type = np.ones(1) if coin.type == agent.type else np.ones(1)*-1
            pos = coin.state.p_pos - agent.state.p_pos
            coin_obs.append(np.concatenate([pos, same_type]))
            dis.append(np.linalg.norm(pos))

        # deal with observation of goals
        num_left_coins = len(world.landmarks)
        indices = np.argsort(dis)
        holder = np.zeros(3)

        coin_obs = [coin_obs[i] for i in indices[:min(num_goal, num_left_coins)]] + [holder] * max(0 ,num_goal - num_left_coins)


        return np.concatenate(self_pos + self_vel + oppo_pos + coin_obs)

    def generate_coin(self,world: World):
        np_random: np.random.Generator = self.np_random

        coins_before = []
        for coin in world.landmarks:
            while self.check_collide(coin, world.agents + coins_before):
                coin.state.p_pos = np_random.uniform(-self.init_bound, +self.init_bound, world.dim_p)
            # coin.type = np_random.integers(0, 2)  # random number of type setting
            coin.color = [np.array([1, 0, 0]), np.array([0, 0, 1])][coin.type]
            coins_before.append(coin)

    def check_collide(self, coin: Landmark, entity: List[Entity]):
        for agent in entity:
            dis = np.sum(np.square(agent.state.p_pos - coin.state.p_pos))
            dis_min = np.square(agent.size + coin.size)
            if dis < dis_min:
                return True
        return False
            

