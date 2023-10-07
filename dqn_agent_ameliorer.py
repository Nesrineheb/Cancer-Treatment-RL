import gym
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # Discount factor (adjust as needed)
        self.epsilon = 1.0  # Exploration rate (adjust as needed)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001  # Learning rate (adjust as needed)
        self.batch_size = 64  # Batch size (adjust as needed)
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.model.predict(np.array([state]))[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)
        
        states = np.array(states)
        targets = np.array(targets)
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    # Créez votre environnement Gym ici
    env = gym.make('CancerGrowth-v0')

    # Obtenez les tailles de l'espace d'observation et d'action de votre environnement
    state_size = env.observation_space.shape[0]  # Taille de l'espace d'observation
    action_size = env.action_space.n  # Nombre d'actions possibles

    # Créez une instance de l'agent DQN
    agent = DQNAgent(state_size, action_size)

    # Entraînez l'agent et collectez des données d'entraînement
    num_episodes = 500  # Remplacez par le nombre d'épisodes souhaité
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            # L'agent choisit une action
            action = agent.act(state)

            # L'environnement effectue l'action et renvoie le nouvel état et la récompense
            next_state, reward, done, _ = env.step(action)

            # L'agent mémorise l'expérience
            agent.remember(state, action, reward, next_state, done)

            # Passage à l'état suivant
            state = next_state
            total_reward += reward

            if done:
                episode_rewards.append(total_reward)
                break

        # Entraînez l'agent avec l'expérience mémorisée
        agent.replay()

        # Affichez les progrès toutes les 100 épisodes
        if (episode + 1) % 100 == 0:
            print(f"Épisode {episode + 1}, Récompense totale : {np.mean(episode_rewards[-100:])}")

    # Tracez la récompense moyenne au fil des épisodes
    plt.plot(episode_rewards)
    plt.xlabel('Épisode')
    plt.ylabel('Récompense totale')
    plt.title('Récompense au fil des épisodes')
    plt.show()

    # Sauvegardez le modèle pré-entraîné
    model_save_path = 'dqn_model'
    agent.model.save(model_save_path)

    env.close()
