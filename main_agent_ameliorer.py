from train import main as train_main
from evaluate import evaluate_model
import gym
from cancer_growth_env import CancerGrowthEnv
from dqn_agent import DQNAgent  # Import your DQN agent class
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    #train_main()  # Exécuter l'entraînement
    model_path = os.path.join('RL_Cancer_Treatment', 'dqn_model')
    evaluate_model(model_path)  # Exécuter l'évaluation du modèle
