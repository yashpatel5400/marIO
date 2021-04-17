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
epsilon_decay = 0.75
epsilon_decay_step = 100
target_model_step = 50
replay_size = 1_00
minibatch_size = 25

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

    if epoch > 0 and epoch % epsilon_decay_step == 0: 
        epsilon *= epsilon_decay

    if epoch > 0 and epoch % target_model_step == 0: 
        target_model = copy.deepcopy(model)

    for _ in range(replay_size):
        if done:
            state = env.reset()

        qs = model(to_tensor(state, use_gpu))
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(from_tensor(qs)[0])

        # buffer holds (state, action, reward, next_state)
        next_state, reward, done, _ = env.step(action=action)
        replay_buffer.append((state, action, reward, next_state))
        env.render()

    random_sample = np.random.random_sample((minibatch_size,)) * replay_size
    random_sample = [int(sample) for sample in random_sample]
    minibatch = [replay_buffer[idx] for idx in random_sample]
    X = torch.stack([model(to_tensor(state, use_gpu))[action] for (state, action, _, _) in minibatch])
    Y = torch.stack([reward + gamma * torch.max(target_model(to_tensor(next_state, use_gpu))) for (_, _, reward, next_state) in minibatch])
    
    loss = loss_fn(X, Y)
    model.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch}/{train_epochs}]: {loss}")

state = env.reset()
for epoch in range(train_epochs):
    if done:
        state = env.reset()

    qs = model(to_tensor(state, use_gpu))
    action = np.argmax(from_tensor(qs)[0])
    state, _, done, _ = env.step(action=action)

    env.render()
