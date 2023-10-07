import numpy as np
import matplotlib.pyplot as plt

class Evaluate:
    def __init__(self, env, agent, num_episodes):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes

    def evaluate(self):
        episode_rewards = []

        for episode in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0

            while True:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state

                if done:
                    break

            episode_rewards.append(total_reward)

        # Calculate and print the mean reward over episodes
        mean_reward = np.mean(episode_rewards)
        print(f"Mean Reward over {self.num_episodes} episodes: {mean_reward}")

        # Plot the rewards
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_episodes + 1), episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Evaluation Rewards')
        plt.show()

        return episode_rewards
