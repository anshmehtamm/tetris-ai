import gymnasium
import numpy as np
import random
from tqdm import tqdm
import pickle
import zlib
import os

# default action space is 5
env = gymnasium.make("ALE/Tetris-v5", obs_type="ram", render_mode="rgb_array")

# if file exists, load it
if (os.path.exists('q_learning.gz')):
    with open('q_learning.gz', 'rb') as fp:
        print("Loading Q Table")
        data = zlib.decompress(fp.read())
        Q_table = pickle.loads(data)
else:
    Q_table = {}


done = True
gamma = 0.9
epsilon = 0.9
eta = 1

for i in tqdm(range(20000)):
    state, info = env.reset()
    state = hash(tuple(state))
    done = False
    total_reward = 0
    while not done:
        if (state not in Q_table):
                Q_table[state] = np.zeros(env.action_space.n)
        # choose an action
        if (random.random()<epsilon):
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state])
        # take the action
        new_state, reward, done, x, info = env.step(action)
        new_state = hash(tuple(new_state))
        # update the q table
        if (new_state not in Q_table):
            Q_table[new_state] = np.zeros(env.action_space.n)
        V_opt = max(Q_table[new_state])
        Q_table[state][action] = (1-eta)*Q_table[state][action] + eta*(reward+gamma*V_opt)
        state = new_state
        total_reward += reward
    print("Episode: ", i, "Score: ", "Total Reward: ",total_reward, "Epsilon: ", epsilon )
    total_reward = 0
    epsilon = 0.99999 * epsilon 
env.close()

# save the q table
# convert the ndarray to list
for key in Q_table:
    if (type(Q_table[key]) == np.ndarray):
        Q_table[key] = Q_table[key].tolist()

with open('q_learning.gz', 'wb') as fp:
  fp.write(zlib.compress(pickle.dumps(Q_table, pickle.HIGHEST_PROTOCOL),9))

# new env with trained q table in human mode

# loading the q table
# with open('test.gz', 'rb') as fp:
#     data = zlib.decompress(fp.read())
#     successDict = pickle.loads(data)

env = gymnasium.make("ALE/Tetris-v5", obs_type="ram", render_mode="human")
done = True
continue_playing = True
episode = 0
while continue_playing:
    done = False
    state, info = env.reset()
    total_reward = 0
    while not done:
        state = hash(tuple(state))
        if (state not in Q_table):
            Q_table[state] = np.zeros(env.action_space.n)
        action = np.argmax(Q_table[state])
        new_state, reward, done, x, info = env.step(action)
        total_reward += reward
        state = new_state
        env.render()
    if (input("Continue Playing? (y/n)") == "n"):
        continue_playing = False
    episode += 1
    print("Episode: ", episode, "Reward: ",total_reward)