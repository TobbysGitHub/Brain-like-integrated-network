import os

import numpy as np
import torch
import gym
import time
from collections import deque

# from agent import Agent
from env.car_racing.agent import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rgb2gray(rgb, norm=True):
    # rgb image -> gray [0, 1]
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray


seed = 0
img_stack = 4
# action_repeat = 10
action_repeat = 1
env = gym.make('CarRacing-v0', verbose=0)
state = env.reset()
reward_threshold = env.spec.reward_threshold


class Wrapper():
    """
    Environment wrapper for CarRacing
    """

    def __init__(self, env):
        self.env = env

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = env.reset()
        img_gray = rgb2gray(img_rgb)
        self.stack = [img_gray] * img_stack  # four frames for decision
        self.direct = []
        return np.array(self.stack)

    def step(self, action):
        self.direct.append(action[0])
        total_reward = 0
        for i in range(action_repeat):
            img_rgb, reward, die, _ = env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == img_stack
        return np.array(self.stack), total_reward, done, die

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


def load(agent, directory, filename):
    agent.net.load_state_dict(torch.load(os.path.join(directory, filename), map_location=device))


def save(data, directory, filename):
    torch.save(data, f=os.path.join(directory, filename))


def record(env_wrap, agent):
    state = env_wrap.reset()

    scores_deque = deque(maxlen=100)

    state = env_wrap.reset()
    score = 0

    time_start = time.time()

    states = []
    actions = []
    rewards = []
    while True:
        states.append(state[0].reshape(-1))
        action, a_logp = agent.select_action(state)
        action = action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
        actions.append(action.reshape(-1))
        # env.render()
        next_state, reward, done, die = env_wrap.step(action)
        rewards.append(np.array([reward]))

        state = next_state
        score += reward

        if done or die:
            break

    s = (int)(time.time() - time_start)

    scores_deque.append(score)

    print('Score: {:.2f} \tTime: {:02}:{:02}:{:02} \tSteps:{}'
          .format(score, s // 3600, s % 3600 // 60, s % 60, len(states)))

    return torch.cat([torch.from_numpy(np.array(x)) for x in (states, actions, rewards)], dim=-1), score


agent = Agent(device)

env_wrap = Wrapper(env)

load(agent, 'dir_chk', 'model_weights_820-980.pth')

n_episodes = 64
min_steps = 700
selected_steps = 512
min_score = 800

results = []
while n_episodes > 0:
    result, score = record(env_wrap, agent)
    if len(result) >= selected_steps and score > min_score:
        n_episodes -= 1
        results.append(result[:selected_steps])
results = torch.stack(results)

save(results, '../data', 'car-racing.64')
pass

if __name__ == '__main__':
    pass
