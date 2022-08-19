import gym
import numpy as np
import random
from time import sleep


def epsilon_greedy(qtable, state, epsilon, environment):
        if random.uniform(0,1) < epsilon:
                return environment.action_space.sample()
        else:
                return np.argmax(qtable[state, :])

def boltzamnn(qtable, state, T):
        p = np.exp(qtable[state, :] / T) / np.sum(np.exp(qtable[state, :] / T))
        probability = 0.0
        choice = random.uniform(0, 1)
        for i, prob in enumerate(p):
                probability += prob
                if probability > choice:
                        return i

def qleraning(environment, epochs, steps,strategy, strategy_param, learning_rate, gamma):
        qtable = np.zeros((environment.observation_space.n, environment.action_space.n))
        for _ in range(epochs):
                state = environment.reset()
                finished = False
                reward = 0
                for _ in range(steps):
                        if strategy == "epsilon_greedy":
                                move = epsilon_greedy(qtable, state, strategy_param, environment)
                        else:
                                move = boltzamnn(qtable, state, strategy_param)
                        new_state, reward, finished, _ = environment.step(move)
                        qtable[state, move] = (1 - learning_rate) * qtable[state, move] + learning_rate * (reward + gamma * np.max(qtable[new_state]))
                        if finished == True: 
                                break
                        state = new_state
        return qtable
        
def test_qleraning(environment, qtable, epochs, visualizing):
        success = 0
        steps = []
        for _ in range(epochs):
                step= 0
                state = environment.reset()
                finished = False
                for _ in range(100):
                        move = np.argmax(qtable[state, :])
                        state, reward, finished, _= environment.step(move)
                        step += 1
                        steps.append(step)
                        if visualizing:
                                environment.render()
                                print(f"Steps: {step}")
                                print(f"Reward: {reward}")
                                sleep(0.2)
                        if finished:
                                if reward == 20:
                                        success += 1
                                break
        print(f"Avarage steps to find solution: {sum(steps) / len(steps)}")
        print(f"Avarage successes: {success / epochs}")

def main():
        environment = gym.make("Taxi-v3").env
        qtable = qleraning(environment=environment, epochs = 2000, steps = 200, strategy = "epsilon_greedy", strategy_param=0.1, learning_rate=0.3, gamma=0.9)
        test_qleraning(environment=environment, qtable=qtable, epochs=1000, visualizing=False)

if __name__ == "__main__":
        main()


