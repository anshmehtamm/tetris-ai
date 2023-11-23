import gymnasium
from tqdm import tqdm

env = gymnasium.make("ALE/Tetris-v5", obs_type="ram", render_mode="human")
# default action space is 5
print("Action Space:", env.action_space.n)
for i in tqdm(range(10)):
    done = False
    state, info = env.reset()
    while not done:
        if (done):
            break
        new_state, reward, done, x, info = env.step(env.action_space.sample())
        env.render()
env.close()