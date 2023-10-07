'''import gym
from cancer_growth_env import CancerGrowthEnv
from rl_agent import RandomAgent  # Import your RL agent class

# Create the environment
env = gym.make('CancerGrowth-v0')

# Create an instance of your RL agent
agent = RandomAgent(env.action_space)

# Training loop
for episode in range(100):
    state = env.reset()
    total_reward = 0

    while True:
        # Use the agent's policy to select an action
        action = agent.select_action(state)
        
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")
            break

env.close()'''
import gym
from cancer_growth_env import CancerGrowthEnv
from dqn_agent import DQNAgent  # Import your DQN agent class
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt


def save_reward_chart(episode_rewards, save_dir='RL_Cancer_Treatment'):
      os.makedirs(save_dir, exist_ok=True)
      plt.plot(episode_rewards)
      plt.xlabel('Épisode')
      plt.ylabel('Récompense totale')
      plt.title('Récompense au fil des épisodes')
      plt.savefig(os.path.join(save_dir, 'reward_chart.png'))
      plt.close()
# Create the environment
env = gym.make('CancerGrowth-v0')

# Create an instance of your DQN agent
state_size = 1  # Update with your observation space size
action_size = 2  # Update with your action space size
agent = DQNAgent(state_size, action_size)

# Training loop
num_episodes = 500
episode_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    while True:
        # Use the DQN agent to select an action
        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward

        # Remember the experience for replay
        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print(f"Épisode {episode + 1}, Récompense totale : {total_reward}")
            episode_rewards.append(total_reward)
            break

    # Train the agent with replay buffer
    agent.replay(batch_size=32)
    # Sauvegardez le modèle pré-entraîné
model_save_path = os.path.join('RL_Cancer_Treatment', 'dqn_model')
agent.model.save(model_save_path)

# Tracez un dernier graphique de récompense
save_reward_chart(episode_rewards)

env.close()

# Plot the rewards over episodes
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward over Episodes')
plt.show()



