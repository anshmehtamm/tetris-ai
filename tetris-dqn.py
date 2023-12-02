import gymnasium
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
from tetris_q_net import TetrisQNet

EPISODES = 10000
EPSILON = 0.99
AGGREGATE_STATS_EVERY = 50
MIN_REWARD = 5

if not os.path.isdir('models'):
    os.makedirs('models')

def pre_process_image(image):
    # crop image
    image = image[26:203, 21:64, :]
    # luma transform
    image = np.dot(image, [0.299, 0.587, 0.114])
    # round to nearest integer
    image = np.round(image)
    return image

env = gymnasium.make("ALE/Tetris-v5", render_mode="rgb_array")
agent = TetrisQNet()


eq_rewards = []
print(env.action_space.n)

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
    agent.tensorboard.step = episode
    episode_reward = 0
    step = 1
    current_state, info = env.reset()
    current_state = pre_process_image(current_state)
    done = False

    while not done:
        if np.random.random() > EPSILON:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.action_space.n)
        
        new_state, reward, done, x, info = env.step(action)
        new_state = pre_process_image(new_state)

        episode_reward += reward
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        #print(f"Episode: {episode}, Step: {step}, Reward: {episode_reward}, Action: {action}, Epsilon: {EPSILON}")
        step += 1
    print(f"Episode: {episode}, Reward: {episode_reward}")
    eq_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(eq_rewards[-AGGREGATE_STATS_EVERY:])/len(eq_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(eq_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(eq_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=EPSILON)

        if max_reward >= MIN_REWARD:
            agent.model.save(f"models/{episode}_model_{max_reward}max_{min_reward}min_{average_reward}_avg.h5")
        
    if EPSILON > 0.05:
        EPSILON *= 0.9999
        EPSILON = max(0.05, EPSILON)