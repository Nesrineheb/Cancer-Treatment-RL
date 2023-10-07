import gym
from gym import spaces
import numpy as np

class CancerGrowthEnv(gym.Env):
   
# Register the environment with Gym
    gym.register(
    id='CancerGrowth-v0',
    entry_point='cancer_growth_env:CancerGrowthEnv',
    )


    def __init__(self):
        super(CancerGrowthEnv, self).__init__()

        # Définissez les espaces d'observation et d'action
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0: Pas de traitement, 1: Traitement

        # Initialisez l'état initial (représentant la taille de la tumeur)
        self.state = 0.1  # Taille initiale de la tumeur

    def step(self, action):
        # Simulation de la croissance tumorale
        growth_rate = 0.1  # Taux de croissance arbitraire
        if action == 1:
            growth_rate -= 0.05  # Effet du traitement

        self.state *= (1 + growth_rate)

        # Récompense (plus la tumeur est petite, meilleure est la récompense)
        reward = -self.state

        # Définir si l'épisode est terminé (tumeur trop grande ou petite)
        done = self.state <= 0.01 or self.state >= 2.0

        return self.state, reward, done, {}

    def reset(self):
        # Réinitialisez l'état de l'environnement
        self.state = 0.1
        return self.state
