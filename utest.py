# -*- coding: utf-8 -*-
"""
@author: Hr
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


lr = RandomForestRegressor()


import gym
env = gym.make('Collab-v0')
actionsObservationsForGame = []
nextObservations = []
for i_episode in range(200):
    observation = env.reset()
    actions = []
    observations = []
    observations.append(observation)
    for t in range(100):
        # env.render()
        #0,1,2,3, g,w,d,j
        # print(observation)
        action = env.action_space.sample()
        actions.append(action)
        # observation : pos, vitesses (8)
        #reward : -5 si rien, +epsilon si hauteur, +bcp si toucher l'autre côté
        # done : true si fini
        observation, reward, done, info = env.step(action)
        observations.append(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    actionsObservations = np.array([np.concatenate([np.array(observations[i+1]), np.array([actions[i]])]) for i in range(len(observations)-1)])
    translatedObservations = np.array([observations[i+1] - observations[i] for i in range(len(observations)-1)])
    # print(translatedObservations)
    # print(observations)
    if i_episode >=1:
        pred = lr.predict(actionsObservations)
        rel_error = 0.
        for i in range(len(pred)):
            error = np.mean([(1-pred[i][j]/translatedObservations[i][j])**2 for j in range(len(pred[i]))])
            rel_error += error
        rel_error = 100*rel_error/len(pred)
        print("Relative error (%)")
        # print(np.mean((lr.predict(actionsObservations) - translatedObservations) ** 2))
   # pred = lr.predict(actionsObservations)
   	# for i in len(pred):
   	#  print(pred[i], nextObservations[i])
    # print(actionsObservationsForGame.shape, nextObservations.shape)	
    # np.random.shuffle(nextObservations)
    lr.fit(actionsObservations, translatedObservations)








