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
discount = 0.8
eps = 0.01

alpha = 0.5

def probabilities(theta, observation):
	result = np.zeros(2*number_of_agents)
	for i in range(2*number_of_agents):
		obs = np.concatenate([observation, np.array([1])])
		result[i] = np.exp(np.dot(theta[i], obs))
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
		observations.append(obs)
		rewards.append(reward)
		if done: 
			print "YEAH"
			break

	return observations, actions, rewards 

def compute_values(rewards):
	out = []
	l = len(rewards) + 1
	for i, r in enumerate(rewards):
		result = 0.
		v = 1.
		for rest in rewards[i:] :
			result += v * rest
			v = v * discount
		result /= (l - i)
		out.append(result)
	return out

def threshold(theta):
	return np.vectorize(lambda x : min(max(x, -5.),5))(theta)

def update_theta(theta, observations, actions, value_samples, rewards):
	new_theta = np.copy(theta)
	total_time = len(actions)
	for i in xrange(total_time-1):
		if not(i % 5):
			prob_array = probabilities(theta, observations[i])
			incr =  alpha * gradient(theta, observations[i], actions[i], prob_array) * (value_samples[i+1] - value_samples[i])
			#print incr
			u = np.random.rand()
			if u>eps:
				new_theta += incr
			else:
				#print 'jump'
				x, y = new_theta.shape
				new_theta += alpha*(np.random.rand(x, y) -0.5)
	return threshold(new_theta)

def gradient(theta, state, instr, prob_array):
	action = instr / 2
	out = np.zeros(theta.shape)
	state_intercept = np.concatenate([state, np.array([1])])
	out[action] = state_intercept * (1 - prob_array[action])
	for i in range(len(out)):
		if not(i == action):
			out[i] =  -state_intercept * prob_array[action]
	return out

def plot3(theta, observation):
	pos = np.linspace(-5,5,1000)
	probaright = []
	probaleft = []
	obs = observation
	for p in pos:
		obs[0] = p
		probaright.append(probabilities(theta, obs)[1])
	plt.plot(pos, probaright)
	plt.show()


render = 0


theta0 = np.zeros((2*number_of_agents, 4*number_of_agents +1))
theta = np.zeros((2*number_of_agents, 4*number_of_agents +1))

theta = 5 *( np.random.rand(2*number_of_agents, 4*number_of_agents +1) - 0.5 )

all_rewards = []
all_rewards0 = []

# theta = np.array([[0., 0., 0., 0., 0.],
# 				  [0., 0., 0., 0., 0.]])

# bad_theta = np.array([[-0.81190277,  1.174109,    0.8655065,  -0.51387351,  1.19355039],
#  					  [-1.6860885,   2.19755091,  1.55254439, -1.69081659, -1.67365105]])
# theta = bad_theta

# other_bad = np.array([[-0.40665346,  2.45277631, -0.64238133, -0.26812841,  1.70700875],
#  [ 0.03392361, -2.48239417,  2.18644514, -1.06744621, -0.29520185]])
# theta = other_bad

other_bad = np.array([[ 1.18343853, -1.1490264,  0.54416218, 1.60447573,  0.66921842],
 [-2.15885663, -0.50253732, -2.13925098, -0.77729358,  1.84061619]])
theta = other_bad


print 'initial theta : '
print theta

max_steps = 200
num_episodes = 1000

for i_episode in tqdm(range(num_episodes)):
    observation = env.reset()
    observations, actions, rewards = generate_episode(theta, max_steps=max_steps)
    observations0, actions0, rewards0 = generate_episode(theta0, max_steps=max_steps)
    values = compute_values(rewards)
    theta = update_theta(theta, observations, actions, values, rewards)
    all_rewards.append(np.sum(rewards))
    all_rewards0.append(np.sum(rewards0))

print 'final theta : '
print theta


render = 1

env.reset()
best_reward = 0

for j in range(max_steps):
	obs, rew, done, info = env.step(2)
	best_reward += rew

render = 0
env.reset()
observations, actions, rewards = generate_episode(theta, max_steps=2*max_steps)
print sum(actions) / (2. * len(actions))
# plt.plot([o[0] for o in observations])

# print all_rewards

plt.plot(best_reward*np.ones(num_episodes), color='y')
plt.plot(all_rewards)
plt.plot(all_rewards0, color='r')
plt.show()
# observation = env.reset()
# plot3(theta, observation)
#plt.show()
