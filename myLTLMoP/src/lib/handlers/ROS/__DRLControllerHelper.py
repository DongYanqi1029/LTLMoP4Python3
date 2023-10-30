from sensor_msgs.msg import LaserScan
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import rospy
import cmath
import math
from abc import ABC, abstractmethod

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class Policy():
    def __init__(self, model_name, model_path):
        self.max_linear_vel = 0.22
        self.max_angular_vel = 2.0

        self.policy = None
        self.model_name = model_name
        self.model_path = model_path

        self.model_path += '/' + self.model_name
        # if 'TD3' in self.model_name:
        self.state_dim = 40 + 4
        self.scan_ob_dim = 40
        self.policy = Actor('td3', self.state_dim, 2, 512).to(device)

        self.policy.load_state_dict(torch.load(self.model_path))
        self.last_action = np.zeros(2)

    def get_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.policy(state).cpu().data.numpy().flatten()
        self.last_action = action

        action = [action[0]*self.max_linear_vel, action[1]*self.max_angular_vel]

        return action

class Observation():
    def __init__(self, scan_dim, pose, target, last_action):
        self.scan_dim = scan_dim
        self.pose = pose
        self.target = target
        self.last_action = last_action
        self.max_laser_value = 2
        self.max_goal_distance = 3

    def get_scan_data(self):
        laser_scan = None
        rospy.logdebug("Waiting for /scan to be READY...")
        while laser_scan is None and not rospy.is_shutdown():
            try:
                laser_scan = rospy.wait_for_message("/scan", LaserScan, timeout=1.0)
                rospy.logdebug("Current /scan READY=>")

            except:
                rospy.logerr("Current /scan not ready yet, retrying for getting laser_scan")
        return laser_scan

    def get_scan_ob(self):
        scan_data = self.get_scan_data()
        scan_ob = []
        laser_data = list(scan_data.ranges)
        # laser_data_left = laser_data[300:]
        # laser_data_right = laser_data[0:61]
        # laser_data = laser_data_left + laser_data_right
        mod = math.ceil(len(laser_data) / self.scan_dim)

        for i, item in enumerate(laser_data):
            if (i % mod == 0):
                # laser_data单位:m
                # distance = item * 100
                distance = item
                if distance == float('Inf') or np.isinf(distance):
                    scan_ob.append(1)
                elif np.isnan(distance):
                    scan_ob.append(0)
                else:
                    scan_ob.append(np.clip(float((distance)/self.max_laser_value), 0, 1))

        return scan_ob

    def get_goal_ob(self):
        relative_x = self.target[0] - self.pose[0]
        relative_y = self.target[1] - self.pose[1]
        coor = complex(relative_x, relative_y)

        # 极坐标
        # 在机器人朝向左侧为正, 右侧为负
        distance, angle = cmath.polar(coor)  # [-pi, pi]
        angle = angle - self.pose[2]
        pi = 3.14
        if (angle < -pi):  # 相当于在左侧
            angle += 2 * pi
        if (angle > pi):  # 相当于在右侧
            angle -= 2 * pi

        rel_distance = float(np.clip((distance / self.max_goal_distance), 0, 1))
        rel_angle = float(angle) / math.pi

        goal_ob = [rel_distance, rel_angle]

        return goal_ob

    def get_state(self):
        scan_ob = self.get_scan_ob()
        goal_ob = self.get_goal_ob()
        last_action_ob = list(self.last_action)

        state = scan_ob + goal_ob + last_action_ob
        state = np.array(state)

        rospy.logdebug("Scan Observations==>" + str(scan_ob))
        rospy.logdebug("Goal Observations(distance, angle)==>" + str(goal_ob))
        rospy.logdebug("Last Action Observations==>" + str(last_action_ob))

        return state

