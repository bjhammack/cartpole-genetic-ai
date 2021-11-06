import argparse
import gym
from gym.wrappers import Monitor
import numpy as np
import torch

from utils import load_model
from models import TwoLayerModel

def play_agent(agent):
    try: #try and exception block because, render hangs if an erorr occurs, we must do env.close to continue working    
        env = gym.make("CartPole-v0")
        env_record = Monitor(env, './video', force=True)
        observation = env_record.reset()
        last_observation = observation
        r=0
        for _ in range(250):
            env_record.render()
            inp = torch.tensor(observation).type('torch.FloatTensor').view(1,-1)
            output_probabilities = agent(inp).detach().numpy()[0]
            action = np.random.choice(range(game_actions), 1, p=output_probabilities).item()
            new_observation, reward, done, info = env_record.step(action)
            r += reward
            observation = new_observation

            if done:
                break

        env_record.close()
        print("Rewards: ",r)

    except Exception as e:
        env_record.close()
        print(e.__doc__)
        print(e.message)


def get_command_line_arguments():
    """
    Get the command line arguments
    :return:(ArgumentParser) The command line arguments as an ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Training script for CartPole-v0 RL model.')
    parser.add_argument('model_path', help='Path to load model state from.', type=str, default='models/model.pt')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_command_line_arguments()
    game_actions = 2
    play_agent(load_model(args.model_path, TwoLayerModel))