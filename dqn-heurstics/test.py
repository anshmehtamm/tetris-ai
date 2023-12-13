import argparse
import torch
import cv2
from tetris import Tetris

def run_test():
    saved_path  = "models/"
    fps  = 20
    block_size = 30
    height = 20
    width = 10

    if torch.cuda.is_available():
        model = torch.load("{}/tetris_1".format(saved_path))
    else:
        model = torch.load("{}/tetris_1".format(saved_path), map_location=lambda storage, loc: storage)
    model.eval()
    env = Tetris(width=width, height=height, block_size=block_size)
    env.reset()
    if torch.cuda.is_available():
        model.cuda()
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action)

        if done:
            break


if __name__ == "__main__":
    run_test()