#!/usr/bin/env python3

import torch
from typing import List
from torch import nn, Tensor
import copy
import numpy as np

from nes_py.wrappers import JoypadSpace
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import random
import keyboard

# Run `tensorboard --logdir=runs` locally
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# To avoid nondeterminism for debugging, let's seed
# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4)
        # self.tanh = nn.Tanh()
        # self.flatten = nn.Flatten()
        # self.linear = nn.Linear(179883, 7)
        self.linear1 = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4, 2)

    def forward(self, x):
        # x = self.conv(x)
        # x = self.tanh(x)
        # x = self.flatten(x)
        # x = self.linear(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def loss_fn(output: List[Tensor], target: List[Tensor]) -> List[Tensor]:
    """
    Effectively computes mean squared error
    - I could not figure out how to get torch to work properly, so this is kind of a hacky solution
    Ideally, I would want to do:
    `return ((Tensor(outputs) - Tensor(targets))**2).mean()`
    ...but when we call loss.backward(), it gives the following error:
    `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`
    - Any idea how to fix this? Please let me know!
    """
    return (output - target) ** 2

def to_tensor(state, gpu):
    # tensor = torch.from_numpy((state.copy().reshape((1, 3, 240, 256)) / 255).astype('float32'))
    tensor = torch.from_numpy((state.copy().reshape((1, 4))).astype('float32'))
    if gpu:
        return tensor.cuda()
    return tensor

def make_environment(env_name="CartPole-v1"):
    # env = gym_super_mario_bros.make('SuperMarioBros-v0')
    # return JoypadSpace(env, SIMPLE_MOVEMENT)
    return gym.make(env_name)

def train(model, total_steps, gpu, learning_rate, gamma, epsilon, headless):
    env = make_environment()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    done = True
    state = None
    loss = nn.MSELoss()

    for step in range(total_steps):
        if done:
            state  = env.reset()    

        # 1: Choose an action

        qs = model(to_tensor(state, gpu))
        action = np.argmax(qs.cpu().detach().numpy()) if random.random() < epsilon else env.action_space.sample()
        
        # 2: Do the action

        next_state, reward, done, _ = env.step(action)
        next_qs = model(to_tensor(next_state, gpu))

        # 3: Compute delta - actual reward minus expected reward
        # - Use this as loss to update our net (which is our q-store)

        values = qs[0, action]
        values_next = reward + gamma * torch.max(next_qs)

        # loss = loss_fn(values, values_next)
        output = loss(values, values_next)
        state = next_state

        # Backpropagation
        optimizer.zero_grad()
        output.backward()
        optimizer.step()

        if not headless:
            env.render()  # Only render one of the games we are running

        writer.add_scalar('loss', output, step)
        if step % print_period == 0:
            print(f"{step} / {total_steps} ================> {output}")
    
    torch.save(model.state_dict(), serialize_path)

def test(model, testing_steps, gpu):
    env = make_environment()
    done = True
    for step in range(testing_steps):
        if done:
            state = env.reset()
            print("Failed")

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
training_steps = 5_000    # how many steps to run before testing
testing_steps = 200       # how many steps to run before testing
learning_rate = 0.01  # LR for NN param update
gamma = 0.9               # discount factor for RL bellman eqn
epsilon = 0.5             # w/ P = epsilon, choose previously thought to be best action, otherwise explore
headless = True           # rendering while training or not

# ====== MAIN STUFF

model = NeuralNetwork()
if gpu:
    model.cuda()

if loading:
    model.load_state_dict(torch.load(serialize_path))

if perform_train:
    train(model, training_steps, gpu, learning_rate, gamma, epsilon, headless)

if loading:
    model.load_state_dict(torch.load(serialize_path))

if trial_run:
    test(model, testing_steps, gpu)