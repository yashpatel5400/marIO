import copy
import gym
import random
import torch
import torch.nn as nn

import numpy as np

class Network(nn.Module):
    def __init__(self):
        # input is [1, 4]
        super(Network, self).__init__()
        self.linear1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def to_tensor(state, use_gpu):
    tensor = torch.from_numpy(state.astype('float32'))
    if use_gpu:
        return tensor.cuda()
    return tensor

def from_tensor(tensor):
    return tensor.cpu().detach().numpy()

use_gpu = True
train_epochs = 500
test_epochs = 100
lr = 0.001
gamma = 0.99
epsilon = 0.99
epsilon_decay_step = 100
target_model_step = 50
replay_size = 25

model = Network()
if use_gpu:
    model.cuda()
target_model = copy.deepcopy(model)

optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
env = gym.make('CartPole-v1')
state = env.reset()
loss_fn = nn.MSELoss()
done = True

for epoch in range(train_epochs):
    replay_buffer = []
    for _ in range(replay_size):
        if done:
            state = env.reset()

        if epoch > 0 and epoch % epsilon_decay_step == 0: 
            epsilon /= 2

        if epoch > 0 and epoch % target_model_step == 0: 
            target_model = copy.deepcopy(model)

        qs = model(to_tensor(state, use_gpu))
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(from_tensor(qs)[0])

        state, reward, done, _ = env.step(action=action)
        next_qs = target_model(to_tensor(state, use_gpu))
        desired = reward + gamma * torch.max(next_qs)
        actual  = qs[action]

        replay_buffer.append((actual, desired))

    sample = random.randint(0, len(replay_buffer) - 1)
    actual, desired = replay_buffer[sample]
    loss = loss_fn(actual, desired)

    model.zero_grad()
    loss.backward()
    optimizer.step()

    env.render()

    print(f"Epoch [{epoch}/{train_epochs}]: {loss}")

state = env.reset()
for epoch in range(train_epochs):
    if done:
        state = env.reset()

    qs = model(to_tensor(state, use_gpu))
    action = np.argmax(from_tensor(qs)[0])
    state, _, _, _ = env.step(action=action)

    env.render()
