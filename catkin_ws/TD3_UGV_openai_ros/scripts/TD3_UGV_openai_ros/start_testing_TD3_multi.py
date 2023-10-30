#!/usr/bin/env python

import copy

import torch
import torch.nn.functional as F
import torch.nn as nn

import gym
import numpy
import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
from itertools import count
import os
import PIL
import sys
from abc import ABC, abstractmethod

class Network(torch.nn.Module, ABC):
    def __init__(self, name, visual=None):
        super(Network, self).__init__()
        self.name = name
        self.visual = visual
        self.iteration = 0

    @abstractmethod
    def forward():
        pass

    def init_weights(n, m):
        if isinstance(m, torch.nn.Linear):
            # --- define weights initialization here (optional) ---
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        # --- define layers here ---
        self.fa1 = nn.Linear(state_size, hidden_size)
        self.fa2 = nn.Linear(hidden_size, hidden_size)
        self.fa3 = nn.Linear(hidden_size, action_size)

        self.apply(super().init_weights)

    def forward(self, states, visualize=False):
        # --- define forward pass here ---
        x1 = torch.relu(self.fa1(states))
        x2 = torch.relu(self.fa2(x1))
        action = torch.tanh(self.fa3(x2))

        # -- define layers to visualize here (optional) ---
        if visualize and self.visual:
            self.visual.update_layers(states, action, [x1, x2], [self.fa1.bias, self.fa2.bias])
        # -- define layers to visualize until here ---
        return action

class Testing:
    def __init__(self, policy_net):
        self.policy = policy_net
    def selectAction(self, state):
        with torch.no_grad():
            return self.policy(state)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rospy.init_node('myrobot_TD3_testing', anonymous=True, log_level=rospy.WARN)


    goals = [(-1.5, -1)]
    goal_index = 0
    goal_num = len(goals)

    os.environ['GOAL_X'] = str(goals[goal_index][0])
    os.environ['GOAL_Y'] = str(goals[goal_index][1])


    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param("myrobot/task_and_robot_environment_name")
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Testing")
    
    model_path = rospy.get_param("/myrobot/model_path")
    scan_dim = rospy.get_param("/myrobot/scan_dim")
    hidden_size = rospy.get_param("/myrobot/hidden_size")
    n_episodes = rospy.get_param("/myrobot/n_episodes")
    n_steps = rospy.get_param("/myrobot/n_steps")

    policy_net = Actor('td3', scan_dim + 4, 2, hidden_size).to(device)
    if os.path.exists(model_path):
        policy_net.load_state_dict(torch.load(model_path))
        policy_net.eval()
    else:
        rospy.logerr("Policy net model Not Found!")
        env.close()
        sys.exit(0)

    testing = Testing(policy_net)

    # Show on screen the actual situation of the robot
    # env.render()
    # for each episode, we test the robot for nsteps
    for i_episode in range(n_episodes) :
        rospy.logwarn("############### START EPISODE => " + str(i_episode + 1))
        cumulated_reward = 0
        done = False
        # Initialize the environment and get first state of the robot
        observation, _ = env.reset()
        # observation = numpy.expand_dims(observation, axis=0)
        rospy.logerr("Env Reset done")
        state = torch.tensor(observation, dtype=torch.float, device=device)
        rospy.logerr("Episode init state obtained")

        os.environ['GOAL_X'] = str(goals[goal_index][0])
        os.environ['GOAL_Y'] = str(goals[goal_index][1])

        for i_step in range(n_steps):
            rospy.logwarn("############### Start Step=>" + str(i_step + 1))
            # Pick an action based on the current state
            action = testing.selectAction(state)

            # Execute the action in the environment and get feedback
            observation, reward, done, info, _ = env.step(numpy.array(action.squeeze().cpu()))
            # observation = numpy.expand_dims(observation, axis=0)
            reward = torch.tensor([reward], device=device)
            cumulated_reward += reward.item()

            if not done:
                next_state = torch.tensor(observation, dtype=torch.float, device=device)
            else:
                goal_index = (goal_index+1)%goal_num
                os.environ['GOAL_X'] = str(goals[goal_index][0])
                os.environ['GOAL_Y'] = str(goals[goal_index][1])

            # Make the algorithm learn based on the results
            rospy.logdebug("# state we were=>" + str(state))
            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward.item()))
            rospy.logwarn("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logdebug("# State in which we will start next step=>" + str(next_state))

            # Move to the next state
            state = next_state

            # if not (done):
            #     rospy.logdebug("NOT DONE")
            # else:
            #     rospy.logdebug("DONE")
            #     break
            rospy.logwarn("############### END Step=>" + str(i_step + 1))

    rospy.logwarn("Cumulated Reward: " + str(cumulated_reward))

    env.close()
    rospy.logerr("############Finished")