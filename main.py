import torch
from torch import nn
from copy import copy

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import random

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(708 * 252, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

def loss_fn(output, target):
    return torch.max((output - target)**2)

def to_tensor(state):
    return torch.from_numpy((state.copy().reshape((1, 3, 240, 256)) / 255).astype('float32'))

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

learning_rate = 0.01
gamma = 0.9

done = True

snapshot = []

model = NeuralNetwork()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for step in range(5000):
    if done:
        prev_state = env.reset()

    epsilon = 0.5

    # w/ P = epsilon, choose previously thought to be best action, otherwise explore
    prev_sample = to_tensor(prev_state)
    prev_qs = model(prev_sample)
    if random.random() < epsilon:
        action = np.argmax(prev_qs)
    else:
        action = env.action_space.sample()
    
    print(action)

    q_values = model(sample)
    state, reward, done, _ = env.step(action)

    # TODO; why is it model w/ theta_i-1? also, this should be the index of the action chosen
    sample = to_tensor(state)
    q_s = model(sample)
    q_s_prime = reward + gamma * prev_qs

    prev_state = state

    loss = loss_fn(q_s, q_s_prime)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    env.render()

env.close()