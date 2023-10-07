import gym
import numpy as np
from dqn_agent import DQNAgent
import os

def save_reward_chart(episode_rewards, save_dir='RL_Cancer_Treatment'):
    os.makedirs(save_dir, exist_ok=True)
    # Code pour tracer et sauvegarder le graphique de récompense

def main():
    # Créez l'environnement
    env = gym.make('CancerGrowth-v0')

    # Créez une instance de votre agent DQN
    state_size = 1  # Mettez à jour avec la taille de l'espace d'observation
    action_size = 2  # Mettez à jour avec la taille de l'espace d'action
    agent = DQNAgent(state_size, action_size)

    # Paramètres d'entraînement
    num_episodes = 500
    episode_rewards = []

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

            # Mémorisez l'expérience pour la relecture
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print(f"Épisode {episode + 1}, Récompense totale : {total_reward}")
                episode_rewards.append(total_reward)
                break

        # Entraînez l'agent avec le tampon de relecture
        #agent.replay(batch_size=32)
        agent.replay()

    # Sauvegardez le modèle pré-entraîné
    model_save_path = os.path.join('RL_Cancer_Treatment', 'dqn_model')
    agent.model.save(model_save_path)

    # Tracez un graphique de récompense
    save_reward_chart(episode_rewards)

    env.close()

if __name__ == "__main__":
    main()
