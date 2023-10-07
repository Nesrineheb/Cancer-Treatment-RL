import gym

# rl_agent.py

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, state):
        return self.action_space.sample()  # Random action for demonstration
