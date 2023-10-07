import gym
from gym import spaces
import numpy as np

class CancerGrowthEnv(gym.Env):
    
    def __init__(self, num_agents):
        
        super(CancerGrowthEnv, self).__init__()

        self.num_agents = num_agents  # Number of agents
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0: Pas de traitement, 1: Traitement

        # Initialize the state for each agent
        self.states = [0.1] * num_agents  # A list containing the initial state for each agent

    def step(self, actions):
        growth_rate = 0.1  # Arbitrary growth rate
        rewards = []

        for i, action in enumerate(actions):
            if action == 1:
                growth_rate -= 0.05  # Effect of treatment

            self.states[i] = self.states[i] * (1 + growth_rate)  # Update state for each agent

            # Reward (the smaller the tumor, the better the reward)
            reward = -self.states[i]
            rewards.append(reward)

        # Define if the episode is done (tumor too large or too small)
        dones = [state <= 0.01 or state >= 2.0 for state in self.states]

        return self.states, rewards, dones, {}  # Return states as a list

    def reset(self):
        # Reset the state of the environment for each agent
        self.states = [0.1] * self.num_agents  # Return the state as a list
        return self.states
