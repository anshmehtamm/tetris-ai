from collections import deque
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn

from dqn import DeepQNetwork
from tetris import Tetris
from matplotlib import pyplot as plt

decay = 0.999
episodes = 0
highest_score=0
MIN_SAMPLES_REQUIRED = 3000
REPLAY_MEMORY_SIZE = 30000

lines_cleared = []
model = DeepQNetwork()

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
env = Tetris(width=10, height=20, block_size=30)
state = env.reset()

replay = deque(maxlen=REPLAY_MEMORY_SIZE)


while episodes < 10000:
    next_steps = env.get_next_states()
    epsilon = 1e-3 + (max(2000 - episodes, 0) * (
            1 - 1e-3) / 2000)
    
    next_actions, next_states = zip(*next_steps.items())
    next_states = torch.stack(next_states)

    model.eval()

    with torch.no_grad():
        predictions = model(next_states)[:, 0]

    # train the model
    model.train()

    if random() < epsilon:
        index = randint(0, len(next_steps) - 1)
    else:
        index = torch.argmax(predictions).item()

    next_state = next_states[index, :]
    action = next_actions[index]

    lines, done = env.step(action, render=False)

    replay.append([state, lines, next_state, done])

    if not done:
        state = next_state
        continue
    else:
        final_score = env.score
        final_tetrominoes = env.tetrominoes
        final_cleared_lines = env.cleared_lines
        state = env.reset()

    if len(replay) < MIN_SAMPLES_REQUIRED:
        # dont train until we have enough samples
        continue

    episodes += 1
    batch_to_train = sample(replay, min(len(replay), 32))

    state_batch, reward_batch, next_state_batch, done_batch = zip(*batch_to_train)
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
    
    opt.zero_grad()
    loss = criterion(q_values, y_batch)
    loss.backward()
    opt.step()

    print("Game {} finished with score {}".format(episodes, final_score))

    if (episodes>2000 and episodes % 100 == 0):
        # continue
        if (input("Continue?")=="n"):
            break

# plot the average reward
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