import gym

from model.policy_gradient import PolicyGradient


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    model = PolicyGradient(env)
    model.train()
    model.make_video()