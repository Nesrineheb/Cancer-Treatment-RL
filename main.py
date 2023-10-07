from train import Train
import gym
from cancer_growth_env import CancerGrowthEnv
from dqn_agent import DQNAgent  # Importez votre classe DQNAgent
import matplotlib.pyplot as plt
from evaluate import Evaluate 

#Train 
if __name__ == "__main__":
    # Créez votre environnement Gym ici (CancerGrowthEnv dans cet exemple)
    gym.envs.register(
        id='CancerGrowth-v0',
        entry_point='cancer_growth_env:CancerGrowthEnv',
    )
    num_agents = 2  # Number of agents
    env = gym.make('CancerGrowth-v0', num_agents=num_agents)

    

    # Obtenez les tailles de l'espace d'observation et d'action de votre environnement
    state_size = env.observation_space.shape[0]  # Taille de l'espace d'observation
    action_size = env.action_space.n  # Nombre d'actions possibles

    # Créez deux instances de l'agent DQN
    agent1 = DQNAgent(state_size, action_size)
    agent2 = DQNAgent(state_size, action_size)

    # Créez une liste d'agents
    agents = [agent1, agent2]

    # Créez une instance de la classe Train avec les agents
    trainer = Train(env, agents, num_episodes=100)

    # Entraînez les agents et collectez des données d'entraînement
    episode_rewards = trainer.train()

    # Tracez les courbes de récompense pour chaque agent
    for i, rewards in enumerate(episode_rewards):
        plt.plot(rewards, label=f'Agent {i+1}')

    plt.xlabel('Épisode')
    plt.ylabel('Récompense totale')
    plt.title('Récompense au fil des épisodes par agent')
    plt.legend()
    plt.show()

    env.close()
#Evaluate

'''if __name__ == "__main__":
    # Create the environment and agent
    gym.envs.register(
        id='CancerGrowth-v0',
        entry_point='cancer_growth_env:CancerGrowthEnv',
        )
    env = gym.make('CancerGrowth-v0', num_agents=2)  # Pass 'num_agents' as a keyword argument
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Load the pretrained model weights
    agent.model.load_weights('common_dqn_model.h5')  # Replace with the path to your pretrained model weights

    # Create an instance of the Evaluate class and specify the number of evaluation episodes
    num_evaluation_episodes = 100  # Adjust the number of evaluation episodes as needed
    evaluator = Evaluate(env, agent, num_evaluation_episodes)

    # Perform the evaluation
    evaluation_results = evaluator.evaluate()'''
