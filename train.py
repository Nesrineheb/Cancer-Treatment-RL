import gym
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from dqn_agent import DQNAgent  # Assurez-vous d'importer votre classe DQNAgent
from cancer_growth_env import CancerGrowthEnv

class Train:
    def __init__(self, env, agents, num_episodes):
        self.env = env
        self.agents = agents
        self.num_episodes = num_episodes

    def train(self):
        episode_rewards = [[] for _ in range(len(self.agents))]  # Une liste de récompenses pour chaque agent

        for episode in range(self.num_episodes):
            print('-------------_______________________________________NUM', episode)
            states = self.env.reset()  # État initial pour chaque agent

            while True:
                actions = [agent.act(state) for agent, state in zip(self.agents, states)]

                next_states, rewards, dones, _ = self.env.step(actions)

                for i, agent in enumerate(self.agents):
                    agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
                    episode_rewards[i].append(rewards[i])

                states = next_states

                if all(dones):
                    break

            for agent in self.agents:
                agent.replay()

            if (episode + 1) % 100 == 0:
                print(f"Épisode {episode + 1}, Récompense totale : {np.mean(episode_rewards[0][-100:])}")

        # Obtenez le modèle pré-entraîné d'un des agents (commun à tous les agents)
        common_model = self.agents[0].model

        # Sauvegardez le modèle pré-entraîné (le modèle commun)
        common_model.save('common_dqn_model.h5')

        # Sauvegardez la courbe de récompense dans un fichier CSV
        reward_data = {'Episode': list(range(1, self.num_episodes + 1))}
        for i, rewards in enumerate(episode_rewards):
            reward_data[f'Agent_{i+1}'] = rewards
        df = pd.DataFrame(reward_data)
        reward_csv_path = 'training_rewards.csv'
        df.to_csv(reward_csv_path, index=False)

        # Tracé des récompenses des deux agents dans la même chart
        plt.figure(figsize=(10, 6))
        for i, rewards in enumerate(episode_rewards):
            plt.plot(range(1, self.num_episodes + 1), rewards, label=f'Agent_{i+1}')
        plt.xlabel('Épisode')
        plt.ylabel('Récompense')
        plt.legend()
        plt.title('Récompenses par épisode (Agents 1 et 2)')
        plt.savefig('training_rewards_plot.png')  # Enregistrez le plot en tant qu'image
        plt.show()  # Affichez le plot

        return episode_rewards
