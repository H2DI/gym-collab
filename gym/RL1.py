# -*- coding: utf-8 -*-
"""
@author: H
"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import gym

env = gym.make("Collab-v0")
env.reset()

number_of_agents = 1

# theta = np.zeros((2 * number_of_agents, 4 * number_of_agents + 1))
discount = 0.99  # discount factor
eps = 0.0  # exploration

alpha = 0.005  # learning rate in step size


""""
    Policy gradient

    Todo: reduce sensitivity to noise by performing the gradient steps on some averaged values? 
""""


def compute_values(rewards, discount=discount):
    """
    out: list of values of length len(rewards)
    """
    out = []
    l = len(rewards) + 1
    for i, r in enumerate(rewards):
        result = 0.0
        v = 1.0
        for rest in rewards[i:]:
            result += v * rest
            v = v * discount
        result /= l - i
        out.append(result)
    return out


class Model:
    def __init__(self, env, theta_init, max_steps=200, alpha=alpha):
        self.env = env
        self.max_steps = max_steps

        self.theta = theta_init

        self.episodes = []
        self.alpha = alpha

    def action_probs(self, theta, state):
        """
        Build transition probabilities from parametrisation
        in: theta array(actions, state_with_intercept)
        out: array of probabilities over actions
        """
        result = np.zeros(4 * number_of_agents)
        state_intercept = np.concatenate([state, np.array([1])])
        for i in range(4 * number_of_agents):
            result[i] = np.exp(np.dot(theta[i], state_intercept))
        return result / np.sum(result)

    def sample_action(self, theta, state):
        # Actions are: (0:left, 1:wait, 2:right, 3:up) * number of agents
        prob_array = self.action_probs(theta, state)
        # assert np.isclose(np.sum(prob_array), 1)
        N_i = len(prob_array)
        return np.random.choice(N_i, p=prob_array)
        # return np.argmax(prob_array)

    def generate_episode(self, theta, training=False, render=False, max_steps=None):
        state = self.env.reset()
        states = [state]
        actions = []
        rewards = [0]
        if max_steps is None:
            max_steps = self.max_steps
        for _ in range(max_steps):
            if render:
                self.env.render()
            if training:
                action = self.sample_action(theta, state)
            else:
                action = np.argmax(self.action_probs(theta, state))
            actions.append(action)
            state, reward, done, info = self.env.step(action)
            states.append(state)
            rewards.append(reward)
            if done:
                # print("YEAH")
                break
        episode = theta, states, actions, rewards
        if training:
            self.episodes.append(episode)
        else:
            return episode

    def _gradient(self, theta, state, instr, prob_array):
        action = instr
        out = np.zeros(theta.shape)
        state_intercept = np.concatenate([state, np.array([1])])
        out[action] = state_intercept * (1 - prob_array[action])
        for i in range(len(out)):
            if not (i == action):
                out[i] = -state_intercept * prob_array[action]
        return out

    def grad_step(self, episode, time_between_value_updates=5):
        theta, states, actions, rewards = episode
        noisy_rewards = np.array(
            rewards
        )  # + np.random.normal(loc=0, scale=0.1, size=len(rewards))
        value_samples = compute_values(noisy_rewards)
        new_theta = np.copy(theta)
        total_time = len(actions)
        for t in range(total_time - 1):
            if not (t % time_between_value_updates):
                prob_array = self.action_probs(theta, states[t])
                incr = (
                    self.alpha
                    * self._gradient(theta, states[t], actions[t], prob_array)
                    * (value_samples[t + 1] - value_samples[t])
                )
                u = np.random.rand()
                if u <= eps:
                    x, y = new_theta.shape
                    new_theta += self.alpha * (np.random.rand(x, y) - 0.5)
                else:
                    new_theta += incr
        return np.clip(new_theta, a_min=-100, a_max=100)

    def update_theta(self):
        self.theta = self.grad_step(self.episodes[-1])

    def run_one_update(self):
        self.generate_episode(self.theta, training=True)
        self.update_theta()

    def train(self, N_episodes):
        for _ in tqdm(range(N_episodes)):
            self.run_one_update()


def plot3(model, states):
    # Plot proba of going right as a function of x
    return


bad_theta = np.array(
    [
        [1.18343853, -1.1490264, 0.54416218, 1.60447573, 0.66921842],
        [1.18343853, -1.1490264, 0.54416218, 1.60447573, 0.66921842],
        [-2.15885663, -0.50253732, -2.13925098, -0.77729358, 1.84061619],
        [1.18343853, -1.1490264, 0.54416218, 1.60447573, 0.66921842],
    ]
)

good_theta = np.array(
    [
        [1 / 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [-1 / 2, 0, 0, 0, 5],
        [0, 0, 0, 0, 0],
    ]
)

# theta = 5 * (np.random.rand(2 * number_of_agents, 4 * number_of_agents + 1) - 0.5)

max_steps = 200

state = env.reset()
benchmark_rew = 0
for _ in range(max_steps):
    if state[0] < 5:
        action = 2
    else:
        action = 0
    state, reward, _, _ = env.step(action)
    benchmark_rew += reward


env.reset()
model = Model(env, good_theta)
_, states, actions, rewards = model.generate_episode(
    good_theta, max_steps=max_steps, render=False
)
# plt.plot(rewards)
benchmark_theta = np.sum(rewards)
# env.viewer.close()

theta = bad_theta
print("initial theta : ", theta)
model = Model(env, theta)
model.train(500)
model.alpha /= 2
model.train(500)
theta = model.theta
print(theta)
env.reset()
model.generate_episode(theta, max_steps=max_steps, render=True)


episodes = model.episodes

final_rewards = [np.sum(episode[3]) for episode in episodes]
plt.axhline(benchmark_theta, label="Benchmark theta", color="green")
plt.axhline(benchmark_rew, label="Benchmark", color="red")
plt.plot(final_rewards, label="Training rewards", color="blue")
plt.legend()
plt.show()


# print(sum(actions) / (2.0 * len(actions)))
# plt.plot([o[0] for o in observations])

# print all_rewards

# plt.plot(best_reward * np.ones(num_episodes), color="y")
# plt.plot(all_rewards)
# plt.plot(all_rewards0, color="r")

# observation = env.reset()
# plot3(theta, observation)
# plt.show()
