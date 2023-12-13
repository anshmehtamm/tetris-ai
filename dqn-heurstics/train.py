from collections import deque
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn

from dqn import DeepQNetwork
from tetris import Tetris
from matplotlib import pyplot as plt

# read file for numbers of lines cleared
lines_cleared = []
model = DeepQNetwork()
env = Tetris(width=10, height=20, block_size=30)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

state = env.reset()
replay_memory = deque(maxlen=30000)
decay = 0.999
epoch = 0
highest_score=0

while epoch < 10000:
    next_steps = env.get_next_states()
    epsilon = 1e-3 + (max(2000 - epoch, 0) * (
            1 - 1e-3) / 2000)
    u = random()
    random_action = u <= epsilon
    next_actions, next_states = zip(*next_steps.items())
    next_states = torch.stack(next_states)
    model.eval()

    with torch.no_grad():
        predictions = model(next_states)[:, 0]
    model.train()
    if random_action:
        index = randint(0, len(next_steps) - 1)
    else:
        index = torch.argmax(predictions).item()

    next_state = next_states[index, :]
    action = next_actions[index]

    reward, done = env.step(action, render=False)

    replay_memory.append([state, reward, next_state, done])
    if done:
        final_score = env.score
        final_tetrominoes = env.tetrominoes
        final_cleared_lines = env.cleared_lines
        state = env.reset()
    else:
        state = next_state
        continue
    if len(replay_memory) < 3000:
        continue
    epoch += 1
    batch = sample(replay_memory, min(len(replay_memory), 32))
    state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
    state_batch = torch.stack(tuple(state for state in state_batch))
    reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
    next_state_batch = torch.stack(tuple(state for state in next_state_batch))

    q_values = model(state_batch)
    model.eval()
    with torch.no_grad():
        next_prediction_batch = model(next_state_batch)
    model.train()

    y_batch = torch.cat(
        tuple(reward if done else reward + decay * prediction for reward, done, prediction in
              zip(reward_batch, done_batch, next_prediction_batch)))[:, None]
    optimizer.zero_grad()
    loss = criterion(q_values, y_batch)
    loss.backward()
    optimizer.step()

    print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
        epoch,
        10000,
        action,
        final_score,
        final_tetrominoes,
        final_cleared_lines))
    lines_cleared.append(final_score)

    if (epoch>2000 and epoch % 100 == 0):
        # continue
        if (input("Continue?")=="n"):
            break

    # if epoch > 0 and final_score>highest_score:
    #     torch.save(model, "{}/tetrisNew_{}".format("saved_models/", epoch))

    # torch.save(model, "{}/tetrisNew".format("saved_models/"))

avg = []
c = 0
ss = 0
for r in lines_cleared:
    ss+=r
    c+=1
    avg.append(ss/c)

plt.plot(avg, color='red')
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("DQN (Replay Memory=30000))")

plt.savefig("DQN30000.png", dpi=1000)