# -*- coding: utf-8 -*-
"""
@author: Hr
"""


import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import gym
env = gym.make('Collab-v0')
env.reset()

number_of_agents = 1

theta = np.zeros((2*number_of_agents, 4*number_of_agents +1))
discount = 0.9

alpha = 0.0005

def probabilities(theta, observation):
	result = np.zeros(2*number_of_agents)
	for i in range(2*number_of_agents):
		obs = np.concatenate([observation, np.array([1])])
		result[i] = np.exp( np.dot(theta[i], obs))
	result /= np.sum(result)
	return result

def sample_action(theta, observation):
	prob_array = probabilities(theta, observation)
	u = np.random.uniform()
	s = 0
	for i in range(len(prob_array)):
		p = prob_array[i]
		s += p
		if u < s:
			return 2*i

def generate_episode(theta, max_steps):
	obs = env.reset()
	observations = [obs]
	actions = []
	rewards = [0]
	done = False
	t = 0
	for t in (range(max_steps)):
		if render:
			env.render()
		t += 1
		action = sample_action(theta, obs)
		actions.append(action)
		obs, reward, done, info = env.step(action)
		rewards.append(reward)
		observations.append(obs)
		if done: 
			print "YEAH"
			break

	return observations, actions, rewards # warning actions est plus petit de 1 : discard les derniers des autres


def compute_values(rewards):
	out = []
	for i, r in enumerate(rewards):
		result = 0.
		v = 1.
		for j, rest in enumerate(rewards[i:]) :
			result += v * rest
			v = v * discount
		out.append(result)
	return out

def update_theta(theta, observations, actions, value_samples):
	new_theta = np.copy(theta)
	total_time = len(actions)

	for i in xrange(total_time):
		prob_array = probabilities(theta, observations[i])
		new_theta += alpha * gradient(new_theta, observations[i], actions[i], prob_array) * value_samples[i]

	return new_theta

def gradient(theta, state, instr, prob_array):
	action = instr / 2
	out = np.zeros(theta.shape)
	state_intercept = np.concatenate([state, np.array([1])])
	out[action] = state_intercept * (1 - prob_array[action])
	for i in range(len(out)):
		if not(i == action):
			out[i] = - state_intercept * prob_array[action]
	return out

def plot3(theta, observation):
	pos = np.linspace(-5,5,1000)
	proba = []
	obs = observation
	for p in pos:
		obs[0] = p
		proba.append(probabilities(theta,obs)[2])
	plt.plot(pos, proba)
	plt.show()


render = 0
theta0 = np.zeros((2*number_of_agents, 4*number_of_agents +1))
theta = np.zeros((2*number_of_agents, 4*number_of_agents +1))
all_rewards = []
all_rewards0 = []

max_steps = 200

for i_episode in tqdm(range(500)):
    observation = env.reset()
    observations, actions, rewards = generate_episode(theta, max_steps=max_steps)
    observations0, actions0, rewards0 = generate_episode(theta0, max_steps=max_steps)
    values = compute_values(rewards)
    theta = update_theta(theta, observations, actions, values)
    all_rewards.append(np.sum(rewards))
    all_rewards0.append(np.sum(rewards0))

render = 1
generate_episode(theta, max_steps=1000)

env.reset()

best_reward = 0

for j in range(max_steps):
	action = 2
	obs, reward, done, info = env.step(action)
	best_reward += reward

# print all_rewards

plt.plot(best_reward*np.ones(1000), color='y')
plt.plot(all_rewards)
plt.plot(all_rewards0, color='r')
plt.show()
observation = env.reset()
plot3(theta, observation)
