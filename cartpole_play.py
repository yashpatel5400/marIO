#!/usr/bin/env python3

import itertools
import os
import torch
from typing import List, NamedTuple, Optional
from torch import nn, Tensor
import copy

import gym

import random
import time

# Run `tensorboard --logdir=runs` locally
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision.transforms as T
from PIL import Image

import matplotlib.pyplot as plt

# After searching online for a while, just could not get this to work
# def seed():
#     random.seed(0)
#     os.environ['PYTHONHASHSEED'] = str(0)
#     np.random.seed(0)
#     torch.manual_seed(0)
#     torch.cuda.manual_seed(0)
#     torch.cuda.manual_seed_all(0) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
# seed()

### HYPERPARAMS ###

load = True
run_gpu = True
run_train = True
run_test = True

test_epochs = 1_000
epochs = 30_000
epsilon = 0.5  # Epsilon-greedy: Chance of choosing random action
gamma = 0.999  # Discount factor of future rewards
lr = 0.0001

# Test : 10/100     Prod: 100/10_000
replay_buffer_capacity = 1_000
batch_size = 500


### DATA ###


# COPIED FROM TUTORIAL
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


# COPIED FROM TUTORIAL
def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


# COPIED FROM TUTORIAL
def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


def explore_environment() -> None:
    env = gym.make("CartPole-v1")
    state = env.reset()
    for i in itertools.count():
        action = i % 2
        state, reward, done, _info = env.step(action)
        # print(reward)  # Reward is 1.0 at every timestep
        if done:
            print(state)
            # print(f"Game ended after {i} steps")  # Ends after 25 steps
            break
        screen = get_screen(env)
        # env.render.shape is (800, 1200, 3)
        # This is (1, 3, 40, 90) (first one is batch dimension)
        print(screen.shape)
        time.sleep(0.01)

    env.close()


def plot_screen(screen) -> None:
    screen = screen[0].permute(1, 2, 0)
    plt.figure()
    plt.imshow(screen)
    plt.show()


def get_and_plot_screen() -> None:
    env = gym.make("CartPole-v1")
    env.reset()
    screen = get_screen(env)
    env.close()
    plot_screen(screen)


class Experience(NamedTuple):
    state: Tensor  # (1, 3, 40, 90)
    action: int
    reward: float
    next_state: Tensor  # (1, 3, 40, 90)
    done: bool


class ReplayBuffer:
    """
    Allows you to randomly sample without replacement from a buffer
    which is periodically updated with new values
    """

    def __init__(self, capacity: int, sample_size: int) -> None:
        self.capacity = capacity
        self.buffer: List[Optional[ReplayBuffer]] = [None] * capacity
        self.write_pos = 0
        self.sample_size = sample_size

    def add(self, experience: Experience) -> None:
        self.buffer[self.write_pos] = experience
        self.write_pos = (self.write_pos + 1) % self.capacity

    def sample(self) -> List[Experience]:
        end = self.capacity
        # If we have not finished filling up the buffer, only sample from what we
        # have filled in so far
        if self.buffer[self.write_pos] is None:
            end = self.write_pos
        return random.sample(self.buffer[:end], self.sample_size)


def test_replay_buffer() -> None:
    rb = ReplayBuffer(capacity=10, sample_size=3)
    for i in range(15):
        rb.add(i)
    print(rb.buffer, rb.write_pos)  # Expect [10,11,12,13,14,5,6,7,8,9], 5
    for i in range(10):
        print(rb.sample())

    rb = ReplayBuffer(capacity=10, sample_size=3)
    for i in range(5):
        rb.add(i)
    print(rb.buffer, rb.write_pos)  # Expect: [0,1,2,3,4,None,None,None,None,None], 5
    for i in range(10):
        print(rb.sample())

    rb = ReplayBuffer(capacity=10, sample_size=5)
    rb.add(88)
    print(rb.sample())  # Should raise exception: not enough vals to sample from yet
# test_replay_buffer(); raise Exception


### MODEL ###

class QModel(nn.Module):
    def __init__(self) -> None:
        super(QModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=5376, out_features=2)

    def forward(self, x: Tensor) -> Tensor:
        """
        state: (1, 3, 40, 90)
        out: (1, 2,)  - 2 possible actions
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        return self.linear(x)


def test_q_model() -> None:
    qmodel = QModel()
    state = np.random.randn(5, 3, 40, 90)
    out = qmodel.forward(Tensor(state.copy()))
    print(out)  # Expect: shape (5, 2)


# test_q_model()


### TRAIN ###


def pick_action(q_s: Tensor, epsilon: float) -> int:
    """
    Picks random action with epsilon-prob, else best action
    - In all cases, returns index of that action
    qs: (2,)
    out: int
    """
    if random.random() < epsilon:
        return random.randint(0, len(q_s) - 1)  # Note: randint is inclusive
    return int(q_s.argmax())


def test_pick_action() -> None:
    epsilon = 0.5  # Adjust epsilon between 0 and 1
    q_s = Tensor([0.0, 0.3, 0.7])
    for i in range(10):
        print(pick_action(q_s=q_s, epsilon=epsilon))


def train(qmodel):

    # Used to predict value of rest of episode (actual Q-value)
    # - We will update it periodically (target_update_period)
    # - This reduces variance
    target_qmodel = None
    target_update_period = 20  # For more complex environments, this can go a lot higher (few thousand)

    loss_fn = nn.MSELoss()
    # # Use huber loss instead - https://en.wikipedia.org/wiki/Huber_loss (did not seem to improve model)
    # loss_fn = nn.functional.smooth_l1_loss

    optimizer = torch.optim.Adam(params=qmodel.parameters(), lr=lr)
    
    steps_taken = 0
    screen = get_screen(env)
    state = screen - screen
    for epoch in range(epochs):

        print(epoch)
        if epoch % target_update_period == 0:
            target_qmodel = copy.deepcopy(qmodel)
        if epochs > 1000 and epoch % 200 == 0:
            print(f"EPOCH {epoch}")
        # plot_screen(state)  # Debug: ensure that the states look like valid differences (they do)
        if run_gpu:
            state = state.cuda()
        q_s = qmodel.forward(x=state).flatten()
        a = pick_action(q_s=q_s, epsilon=epsilon)

        # Take a step
        _state, reward, done, _info = env.step(action=a)

        next_screen = get_screen(env)
        next_state = next_screen - screen
        if run_gpu:
            next_state = next_state.cuda()

        replay_buffer.add(experience=Experience(state=state, action=a, reward=reward, next_state=next_state, done=done))
        if epoch > batch_size:  # Wait until we have enough experiences for a full batch
            experiences: List[Experience] = replay_buffer.sample()
            states, actions, rewards, next_states, dones = map(list, zip(*experiences))
            states = torch.cat(states)  # Shape: (batch_size, 3, 40, 90)
            next_states = torch.cat(next_states)  # Shape: (batch_size, 3, 40, 90)

            # Calculate expected Q-values
            q_s = qmodel.forward(x=states)  # (batch_size, num_actions)
            q_sa_old = q_s[range(batch_size), actions]

            # Calculate target Q-values
            q_sp = target_qmodel.forward(x=next_states).max(axis=1).values  # Shape (batch_size,)
            # Value is zero when state is terminal
            if run_gpu:
                q_sp = torch.where(Tensor(dones).to(bool), torch.zeros_like(q_sp).cpu(), q_sp.cpu())
            else:
                q_sp = torch.where(Tensor(dones).to(bool), torch.zeros_like(q_sp), q_sp)
            q_sa_new = Tensor(rewards) + gamma * q_sp

            if run_gpu:
                loss = loss_fn(q_sa_old.cpu(), q_sa_new.cpu())
            else:
                loss = loss_fn(q_sa_old, q_sa_new)

            writer.add_scalar("loss", loss, epoch)
            qmodel.zero_grad()
            loss.backward()
            # # Grad clamp (did not seem to improve model)
            # for param in qmodel.parameters():
            #     param.grad.data.clamp_(-1, 1)
            optimizer.step()

        steps_taken += 1
        if done:
            writer.add_scalar("steps_taken", steps_taken, epoch)
            # print(f"Steps taken: {steps_taken}")
            steps_taken = 0

        if done:
            env.reset()
            screen = get_screen(env)
            state = screen - screen
        else:
            screen = next_screen
            state = next_state

        # print(f"Epoch: {epoch}\tLoss:{loss}")

    torch.save(qmodel.state_dict(), "dummy")


def test():
    env.reset()
    screen = get_screen(env)
    state = screen - screen

    survivals = []
    cur_run_len = 0
    for epoch in range(test_epochs):
        if run_gpu:
            state = state.cuda()
        q_s = qmodel.forward(x=state).flatten()
        a = pick_action(q_s=q_s, epsilon=-1.0) # no longer want random

        # Take a step
        _, _, done, _ = env.step(action=a)

        next_screen = get_screen(env)
        next_state = next_screen - screen
        cur_run_len += 1

        if done:
            env.reset()
            screen = get_screen(env)
            state = screen - screen
            survivals.append(cur_run_len)
            cur_run_len = 0
        else:
            screen = next_screen
            state = next_state
    survivals.append(cur_run_len)
    print(f"Average: {np.mean(survivals)}")

# test_pick_action()


# TRAIN
# - Good to have inline, so I can cut off training while still having all the variables in my environment

writer = SummaryWriter(comment='_CartPole_target_period_20_lr_0001_batchnorm_huber_loss_target_update_200_grad_clamp')

replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity, sample_size=batch_size)

# Set up data-generator
env = gym.make('CartPole-v1')
env.reset()

# Set up model
qmodel = QModel()  # Used to act and generate expected Q values

if load:
    qmodel.load_state_dict(torch.load("dummy"))

if run_gpu:
    qmodel.cuda()

if run_train:
    train(qmodel)

if run_test:
    test()

env.close()