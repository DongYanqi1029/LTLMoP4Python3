import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 512)
		self.l2 = nn.Linear(512, 256)
		self.l3 = nn.Linear(256, 64)
		self.l4 = nn.Linear(64, action_dim)
		
		self.max_action = torch.from_numpy(max_action).to(device)
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = F.relu(self.l3(a))
		return self.max_action * torch.tanh(self.l4(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 512)
		self.l2 = nn.Linear(512, 256)
		self.l3 = nn.Linear(256, 64)
		self.l4 = nn.Linear(64, 1)

		# Q2 architecture
		self.l5 = nn.Linear(state_dim + action_dim, 512)
		self.l6 = nn.Linear(512, 256)
		self.l7 = nn.Linear(256, 64)
		self.l8 = nn.Linear(64, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = F.relu(self.l3(q1))
		q1 = self.l4(q1)

		q2 = F.relu(self.l5(sa))
		q2 = F.relu(self.l6(q2))
		q2 = F.relu(self.l7(q2))
		q2 = self.l8(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = F.relu(self.l3(q1))
		q1 = self.l4(q1)
		return q1

class PPOActor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(PPOActor, self).__init__()
		self.max_action = max_action
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(state_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(), 
			nn.Linear(128, action_dim),
			nn.Tanh()
		)
		self._initialize_weights()

	def forward(self, x):
		x = self.linear_relu_stack(x)
		result = x * self.max_action.item()
		return result

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)  #正态分布赋值 
				nn.init.constant_(m.bias, 0)

class PPOCritic(nn.Module):
    def __init__(self, state_dim):
        super(PPOCritic, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Linear(128, 1),
        )
        self._initialize_weights()

    def forward(self, x):
        result = self.linear_relu_stack(x)
        return result

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  #正态分布赋值 
                nn.init.constant_(m.bias, 0)