from model import DeepQNetwork, Agent
from utils import plotLearning
import numpy as np

if __name__ == '__main__':
    agent = Agent(gamma=0.95, epsilon=1.0,
                  alpha=0.003, maxMemorySize=5000,
                  replace=None)

    while agent.memCntr < agent.memSize:
        # TODO get observation
        done = False
        while not done:
            # TODO get action
            # TODO get observation_, reward, done, info
            '''
            if done and loss:
                reward -= 100
            '''
            agent.storeTransition(observation, action, reward,
                                  observation)
            observation = observation_

    print('done initializing memory')

    scores = []
    epsHistory = []
    numGames = 50
    batch_size = 32

    for i in range(numGames):
        print('starting game ', i+1, 'epsilon: %.4f'%agent.EPSILON)
        epsHistory.append(agent.EPSILON)
        done = False
        # TODO obs = env.reset()
        # TODO frames
        scores = 0
        lastAction = 0

        while not done:
            if len(frames) == 3:
                action = brain.chooseAction(frames)
                frames = []
            else:
                action = lastAction

            '''
            observation_, reward, done, info = env.step(action)
            score += reward
            frames.append(observation)
            if done and loss:
                reward -= 100
            '''
            agent.storeTransition(observation, action, reward,
                                  observation)
            observation = observation_
            agent.learn(batch_size)
            lastAction = action

        scores.append(score)
        print('score: ',score)
        x = [i + 1 for i in range(numGames)]
        fileName = 'test' + str(numGames) + '.png'
        plotLearning(x, scores, epsHistory, fileName)
