import argparse
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from gym.wrappers import Monitor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy

from models import TwoLayerModel


def init_weights(model):
    if ((type(model) == nn.Linear) | (type(model) == nn.Conv2d)):
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0.00)


def return_random_agents(num_agents):
    agents = []
    for _ in range(num_agents):
        agent = TwoLayerModel()
        for param in agent.parameters():
            param.requires_grad = False
        init_weights(agent)
        agents.append(agent)
    return agents


def run_agents(agents):
    reward_agents = []
    env = gym.make("CartPole-v0")
    for agent in agents:
        agent.eval()
        observation = env.reset()
        r=0
        for _ in range(250):
            inp = torch.tensor(observation).type('torch.FloatTensor').view(1,-1)
            output_probabilities = agent(inp).detach().numpy()[0]
            action = np.random.choice(range(game_actions), 1, p=output_probabilities).item()
            observation, reward, done, info = env.step(action)
            r=r+reward
            if(done):
                break
        reward_agents.append(r)
    return reward_agents


def return_average_score(agent, runs):
    score = 0.
    for i in range(runs):
        score += run_agents([agent])[0]
    return score/runs


def run_agents_n_times(agents, runs):
    avg_score = []
    for agent in agents:
        avg_score.append(return_average_score(agent,runs))
    return avg_score


def mutate(agent):
    child_agent = copy.deepcopy(agent)
    mutation_power = 0.02 # hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
    for param in child_agent.parameters():
        if len(param.shape)==4: # weights of Conv2D
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    for i2 in range(param.shape[2]):
                        for i3 in range(param.shape[3]):
                            param[i0][i1][i2][i3]+= mutation_power * np.random.randn()
        elif len(param.shape)==2: # weights of linear layer
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    param[i0][i1]+= mutation_power * np.random.randn()
        elif len(param.shape)==1: # biases of linear layer or conv layer
            for i0 in range(param.shape[0]):
                param[i0]+=mutation_power * np.random.randn()
    return child_agent


def return_children(agents, sorted_parent_indexes, elite_index):
    children_agents = []
    for i in range(len(agents)-1):
        selected_agent_index = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
        children_agents.append(mutate(agents[selected_agent_index]))
    # one elite child added to next gen
    elite_child = add_elite(agents, sorted_parent_indexes, elite_index)
    children_agents.append(elite_child)
    elite_index=len(children_agents)-1
    
    return children_agents, elite_index


def add_elite(agents, sorted_parent_indexes, elite_index=None, only_consider_top_n=10):
    candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]
    if elite_index is not None:
        candidate_elite_index = np.append(candidate_elite_index,[elite_index])
    top_score = None
    top_elite_index = None
    for i in candidate_elite_index:
        score = return_average_score(agents[i],runs=5)
        if top_score is None:
            top_score = score
            top_elite_index = i
        elif score > top_score:
            top_score = score
            top_elite_index = i
    child_agent = copy.deepcopy(agents[top_elite_index])
    return child_agent


def train(population, generations, top_limit, save_path):
    torch.set_grad_enabled(False)
    agents = return_random_agents(population)
    elite_index = None

    for generation in tqdm(range(generations)):
        rewards = run_agents_n_times(agents, 3)
        sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit] #reverses and gives top values
        top_rewards = []
        for best_parent in sorted_parent_indexes:
            top_rewards.append(rewards[best_parent])
        children_agents, elite_index = return_children(agents, sorted_parent_indexes, elite_index)
        agents = children_agents
        if generation % 50 == 0:
            print(f'\n\nGENERATION: {generation}\nTop Rewards: {[round(i,1) for i in top_rewards][:10]}')

    # Get final agent
    final_rewards = run_agents_n_times(agents, 3)
    sorted_final_indexes = np.argsort(final_rewards)[::-1][:top_limit]
    top_agent = agents[sorted_final_indexes[0]]
    torch.save(top_agent.state_dict(), save_path)


def get_command_line_arguments():
    """
    Get the command line arguments
    return:
        args: The command line arguments as an ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Training script for CartPole-v0 RL model.')
    parser.add_argument('--population', help='Total number of agents per generation', type=int, default=100)
    parser.add_argument('--generations', help='Total number of generations', type=int, default=100)
    parser.add_argument('--gen_turnover', help='Number of top performers to use for next generation', type=int, default=20)
    parser.add_argument('--elite_children', help='Number of children who are copies of top last gen', type=int, default=1)
    parser.add_argument('--save_path', help='Path to save best performer at', type=str, default='saved_models/model.pt')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_command_line_arguments()
    game_actions = 2
    train(args.population, args.generations, args.gen_turnover, args.save_path)