"""Small CPU-friendly provided policy; the learning algorithms are student work."""
import numpy as np
import torch
from torch import nn
from .domains import spec

class Policy(nn.Module):
    def __init__(self, domain, hidden=64):
        super().__init__()
        self.spec = spec(domain)
        self.hidden = hidden
        self.register_buffer("mean", torch.zeros(self.spec.obs_dim))
        self.register_buffer("scale", torch.ones(self.spec.obs_dim))
        self.net = nn.Sequential(nn.Linear(self.spec.obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, self.spec.action_dim))

    def set_normalizer(self, train_obs):
        if self.spec.name.startswith("lander"):
            self.mean.copy_(torch.as_tensor(train_obs.mean(0), dtype=torch.float32))
            self.scale.copy_(torch.as_tensor(train_obs.std(0).clip(.05), dtype=torch.float32))

    def forward(self, obs):
        y = self.net((obs - self.mean) / self.scale)
        return torch.tanh(y) if self.spec.continuous else y

    @torch.no_grad()
    def act(self, obs, mask):
        self.eval()
        output = self(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0]
        if self.spec.continuous:
            return output.cpu().numpy()
        if not np.any(mask):
            raise ValueError("No permitted actions")
        return int(output.masked_fill(~torch.as_tensor(mask, dtype=torch.bool), -torch.inf).argmax())

    def save(self, path):
        torch.save({"domain": self.spec.name, "hidden": self.hidden, "state_dict": self.state_dict()}, path)


def load_policy(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    p = Policy(payload["domain"], payload["hidden"])
    p.load_state_dict(payload["state_dict"])
    return p
