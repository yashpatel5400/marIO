#!/usr/bin/env python3

import torch
from typing import List
from torch import nn, Tensor
import copy
import numpy as np

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import random
import keyboard

# Run `tensorboard --logdir=runs` locally
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


def loss_fn(outputs: List[Tensor], targets: List[Tensor]) -> List[Tensor]:
    """
    Effectively computes mean squared error
    - I could not figure out how to get torch to work properly, so this is kind of a hacky solution
    Ideally, I would want to do:
    `return ((Tensor(outputs) - Tensor(targets))**2).mean()`
    ...but when we call loss.backward(), it gives the following error:
    `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`
    - Any idea how to fix this? Please let me know!
    """
    return [(output - target) ** 2 / len(outputs) for output, target in zip(outputs, targets)]

def to_tensor(state, gpu):
    tensor = torch.from_numpy((state.copy().reshape((1, 3, 240, 256)) / 255).astype('float32'))
    if gpu:
        return tensor.cuda()
    return tensor

def make_environment():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    return JoypadSpace(env, SIMPLE_MOVEMENT)

def train(model, total_steps, gpu, env, learning_rate, gamma, epsilon, headless):
    envs = [make_environment() for _i in range(batch_size)]

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    dones = [True for _ in range(batch_size)]
    states = [None for _ in range(batch_size)]
    for step in range(total_steps):
        for i, done in enumerate(dones):
            if done or keyboard.is_pressed('r'):
                states[i] = envs[i].reset()
        if keyboard.is_pressed('b'):
            break

        if keyboard.is_pressed('a'):
            epsilon -= 0.05
        elif keyboard.is_pressed('d'):
            epsilon += 0.05
        epsilon = max(min(epsilon, 1.0), 0.0)

        # 1: Choose an action

        qs_ls = [model(to_tensor(state, gpu)) for state in states]
        actions = [
            np.argmax(qs.cpu().detach().numpy()) if random.random() < epsilon else env.action_space.sample()
            for env, qs in zip(envs, qs_ls)
        ]

        # 2: Do the action

        next_states, rewards, dones, _ = zip(*[env.step(action) for env, action in zip(envs, actions)])
        next_qs_ls = [model(to_tensor(next_state, gpu)) for next_state in next_states]

        # 3: Compute delta - actual reward minus expected reward
        # - Use this as loss to update our net (which is our q-store)

        values = [qs[0, action] for qs, action in zip(qs_ls, actions)]
        values_next = [reward + gamma * torch.max(next_qs) for next_qs, reward in zip(next_qs_ls, rewards)]

        # if step % print_period == 0:
        #     print(values)
        #     print(values_next)
        # convert `values` and `values_next` from list of tensors to tensor
        losses = loss_fn(values, values_next)

        states = list(next_states)

        # Backpropagation
        optimizer.zero_grad()
        [loss.backward() for loss in losses]
        optimizer.step()

        if not headless:
            envs[0].render()  # Only render one of the games we are running

        writer.add_scalar('loss', sum(losses), step)
        if step % print_period == 0:
            print(f"{step} / {total_steps} ================> {sum(losses)}")

    torch.save(model.state_dict(), serialize_path)

def test(model, testing_steps, gpu, env):
    env = make_environment()
    done = True
    for step in range(testing_steps):
        if done:
            state = env.reset()

        prediction = model(to_tensor(state, gpu))
        qs = prediction.cpu().detach().numpy()
        
        action = np.argmax(qs)
        state, _, done, _ = env.step(action)
        env.render()

    env.close()

"""
from: https://livebook.manning.com/concept/deep-learning/super-mario-bros
actions (7): 'NOOP', 'right', 'right A', 'right B', 'right A B', 'A', 'left'
"""

gpu = True                      # whether to run on GPU
loading = False                 # whether to load whatever's on disk
perform_train = True            # whether to train from the current state (either vanilla or whatever's loaded)
trial_run = True                # whether to do test run
serialize_path = "test.weights" # where to load/save from/to
print_period = 20  # How many steps to print output


# ====== hyperparameters
# How many examples per batch. In practice, we simultaneously run `batch_size` instances of the game
batch_size = 25
training_steps = 1_000    # how many steps to run before testing
testing_steps = 500       # how many steps to run before testing
learning_rate = 0.000001  # LR for NN param update
gamma = 0.9               # discount factor for RL bellman eqn
epsilon = 0.5             # w/ P = epsilon, choose previously thought to be best action, otherwise explore
headless = True           # rendering while training or not

# ====== MAIN STUFF

model = NeuralNetwork()
if gpu:
    model.cuda()

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

if loading:
    model.load_state_dict(torch.load(serialize_path))

if perform_train:
    train(model, training_steps, gpu, env, learning_rate, gamma, epsilon, headless)

if trial_run:
    test(model, testing_steps, gpu, env)