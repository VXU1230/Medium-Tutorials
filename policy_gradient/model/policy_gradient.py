import numpy as np
import os

import tensorflow as tf
from gym import wrappers
import matplotlib.pyplot as plt

from model.baseline_net import BaselineNet
from model.policy_net import PolicyNet


def export_plot(ys, ylabel, title, filename):
    plt.figure()
    plt.plot(range(len(ys)), ys)
    plt.xlabel("Training Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


class PolicyGradient(object):
    def __init__(self, env, num_iterations=300, batch_size=2000, max_ep_len=200, output_path="../results/"):
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.env = env
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.gamma = 0.99
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-2)
        self.policy_net = PolicyNet(input_size=self.observation_dim, output_size=self.action_dim)
        self.baseline_net = BaselineNet(input_size=self.observation_dim, output_size=1)

    def play_games(self, env=None, num_episodes = None):
        episode = 0
        episode_rewards = []
        paths = []
        t = 0
        if not env:
            env = self.env

        while (num_episodes or t < self.batch_size):
            state = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0

            for step in range(self.max_ep_len):
                states.append(state)
                action = self.policy_net.sampel_action(np.atleast_2d(state))[0]
                state, reward, done, _ = env.step(action)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1

                if (done or step == self.max_ep_len-1):
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.batch_size:
                    break

            path = {"observation": np.array(states),
                    "reward": np.array(rewards),
                    "action": np.array(actions)}
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break
        return paths, episode_rewards

    def get_returns(self, paths):
        all_returns = []
        for path in paths:
            rewards = path["reward"]
            returns = []
            reversed_rewards = np.flip(rewards,0)
            g_t = 0
            for r in reversed_rewards:
                g_t = r + self.gamma*g_t
                returns.insert(0, g_t)
            all_returns.append(returns)
        returns = np.concatenate(all_returns)
        return returns

    def get_advantage(self, returns, observations):
        values = self.baseline_net.forward(observations).numpy()
        advantages = returns - values
        advantages = (advantages-np.mean(advantages)) / np.sqrt(np.sum(advantages**2))
        return advantages

    def update_policy(self, observations, actions, advantages):
        observations = tf.convert_to_tensor(observations)
        actions = tf.convert_to_tensor(actions)
        advantages = tf.convert_to_tensor(advantages)
        with tf.GradientTape() as tape:
            log_prob = self.policy_net.action_distribution(observations).log_prob(actions)
            loss = -tf.math.reduce_mean(log_prob * tf.cast(advantages, tf.float32))
        grads = tape.gradient(loss, self.policy_net.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.model.trainable_weights))

    def train(self):
        all_total_rewards = []
        averaged_total_rewards = []
        for t in range(self.num_iterations):
            paths, total_rewards = self.play_games()
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            returns = self.get_returns(paths)
            advantages = self.get_advantage(returns, observations)
            self.baseline_net.update(observations=observations, target=returns)
            self.update_policy(observations, actions, advantages)
            avg_reward = np.mean(total_rewards)
            averaged_total_rewards.append(avg_reward)
            print("Average reward for batch {}: {:04.2f}".format(t,avg_reward))
        print("Training complete")
        np.save(self.output_path+ "rewards.npy", averaged_total_rewards)
        export_plot(averaged_total_rewards, "Reward", "CartPole-v0", self.output_path + "rewards.png")

    def eval(self, env, num_episodes=1):
        paths, rewards = self.play_games(env, num_episodes)
        avg_reward = np.mean(rewards)
        print("Average eval reward: {:04.2f}".format(avg_reward))
        return avg_reward

    def make_video(self):
        env = wrappers.Monitor(self.env, self.output_path+"videos", force=True)
        self.eval(env=env, num_episodes=1)
