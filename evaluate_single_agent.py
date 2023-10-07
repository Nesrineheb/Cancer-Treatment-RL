import gym
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from dqn_agent import DQNAgent  # Assurez-vous d'importer votre classe DQNAgent

def evaluate_model(model_path, num_episodes=2000):
    # Create the environment
    env = gym.make('CancerGrowth-v0')

    # Create an instance of your DQN agent
    state_size = 1  # Update with your observation space size
    action_size = 2  # Update with your action space size
    agent = DQNAgent(state_size, action_size)

    # Load the pretrained model
    agent.model.load_weights(model_path)

    # Evaluation loop
    episode_rewards = []  # Liste pour enregistrer les récompenses de chaque épisode

    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        while True:
            # Utilisez l'agent DQN pour sélectionner une action
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            total_reward += reward

            state = next_state

            if done:
                print(f"Épisode {episode + 1}, Récompense totale : {total_reward}")
                episode_rewards.append(total_reward)

                break

    # Calcul de la moyenne et d'autres métriques
    mean_reward = np.mean(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"Récompense moyenne sur {num_episodes} épisodes : {mean_reward}")
    print(f"Récompense minimale : {min_reward}")
    print(f"Récompense maximale : {max_reward}")
    print(f"Écart-type des récompenses : {std_reward}")

    # Créez un DataFrame pandas pour les récompenses
    rewards_df = pd.DataFrame({'Épisode': range(1, num_episodes + 1), 'Récompense totale': episode_rewards})

    # Enregistrez le tableau des récompenses au format CSV dans le dossier du projet
    rewards_df.to_csv('rewards.csv', index=False)

    env.close()

if __name__ == "__main__":
    model_path = 'RL_Cancer_Treatment/dqn_model'  # Mettez le chemin vers votre modèle pré-entraîné ici
    num_episodes = 500  # Définissez le nombre d'épisodes souhaité
    evaluate_model(model_path, num_episodes)
