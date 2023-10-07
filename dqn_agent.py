import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD  # Utilisation de la descente de gradient (SGD)
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # Facteur de remise (ajustez au besoin)
        self.epsilon = 0.5  # Taux d'exploration initial plus faible
        self.epsilon_min = 0.1  # Valeur minimale de epsilon
        self.epsilon_decay = 0.995  # Taux de décroissance plus lent
        self.learning_rate = 0.001  # Taux d'apprentissage (ajustez au besoin)
        self.batch_size = 64  # Taille du batch (ajustez au besoin)
        self.model = self._build_model()
        self.target_model = self._build_model()  # Ajout du modèle cible

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=SGD(learning_rate=self.learning_rate))  # Utilisation de la descente de gradient
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
                # Utilisation du modèle cible pour évaluer les actions
                next_action = np.argmax(self.model.predict(np.array([next_state]))[0])
                target = reward + self.gamma * self.target_model.predict(np.array([next_state]))[0][next_action]
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)

        states = np.array(states)
        targets = np.array(targets)
        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
