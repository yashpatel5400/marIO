#!/usr/bin/env python3

import torch
from torch import nn
import copy
import numpy as np

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import random
import keyboard

from torch.utils.tensorboard import SummaryWriter
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4)
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(179883, 7)

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

def loss_fn(output, target):
    return (output - target)**2

def to_tensor(state):
    return torch.from_numpy((state.copy().reshape((1, 3, 240, 256)) / 255).astype('float32'))#.cuda()

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

"""
from: https://livebook.manning.com/concept/deep-learning/super-mario-bros
actions (7): 'NOOP', 'right', 'right A', 'right B', 'right A B', 'A', 'left'
"""

# ====== hyperparameters
total_steps = 3000    # how many steps to run before testing
learning_rate = 0.01   # LR for NN param upate
gamma = 0.9            # discount factor for RL bellman eqn
epsilon = 0.5          # w/ P = epsilon, choose previously thought to be best action, otherwise explore
headless = True        # rendering while training or not

loading = False
serialize_path = "test.weights"

# ====== MAIN STUFF
model = NeuralNetwork()
# model.cuda()

if loading:
    model.load_state_dict(torch.load(serialize_path))
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    done = True
    for step in range(total_steps):
        if done or keyboard.is_pressed('r'):
            state = env.reset()
        if keyboard.is_pressed('b'):
            break

        if keyboard.is_pressed('a'):
            epsilon -= 0.05
        elif keyboard.is_pressed('d'):
            epsilon += 0.05
        epsilon = max(min(epsilon, 1.0), 0.0)

        qs = model(to_tensor(state))
        if random.random() < epsilon:
            action = np.argmax(qs.cpu().detach().numpy())
        else:
            action = env.action_space.sample()
        
        next_state, reward, done, _ = env.step(action)
        next_qs = model(to_tensor(next_state))
        
        value = qs[0, action]
        value_next = reward + gamma * torch.max(next_qs)
        
        print(value)
        print(value_next)
        loss = loss_fn(value, value_next)

        state = copy.deepcopy(next_state)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not headless:
            env.render()

        writer.add_scalar('loss', loss, step)
        if step % 100:
            print(f"{step} / {total_steps} ================> {loss}")

    # trial run
    done = True
    for step in range(500):
        if done:
            state = env.reset()
        qs = model(to_tensor(state)).detach().numpy()
        print(qs)
        action = np.argmax(qs)
        state, _, done, _ = env.step(action)
        env.render()

    torch.save(model.state_dict(), serialize_path)

env.close()
