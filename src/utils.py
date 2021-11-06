import torch


def load_model(path, Model):
    agent = Model()
    agent.load_state_dict(torch.load(path))
    agent.eval()
    return agent