from TD3_UGV_openai_ros import networks
from sensor_msgs.msg import LaserScan
import numpy as np
import torch
import rospy
import cmath

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy():
    def __init__(self, policy_name, model_path):
        self.policy = None
        self.policy_name = policy_name
        self.model_path = model_path

        if self.policy_name == "TD3":
            self.model_path += '/TD3_MyRobotWorld-v0_actor'
            self.policy = networks.Actor(36+2+1, 1, np.array([0.3])).to(device)

        self.policy.load_state_dict(torch.load(self.model_path))
        self.last_action = np.zeros(1)

    def get_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.policy(state).cpu().data.numpy().flatten()
        self.last_action = action
        return action

class Observation():
    def __init__(self, scan_dim, pose, target, last_action):
        self.scan_dim = scan_dim
        self.pose = pose
        self.target = target
        self.last_action = last_action
        self.max_laser_value = 6
        self.min_laser_value = 0

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
        mod = (len(laser_data) // self.scan_dim)

        for i, item in enumerate(laser_data):
            if (i % mod == 0):
                # laser_data单位:m
                # distance = item * 100
                distance = item
                if distance == float('Inf') or np.isinf(distance):
                    scan_ob.append(self.max_laser_value)
                elif np.isnan(distance):
                    scan_ob.append(self.min_laser_value)
                else:
                    scan_ob.append(distance)

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

        goal_ob = [distance, angle]

        return goal_ob

    def get_ob(self):
        scan_ob = self.get_scan_ob()
        goal_ob = self.get_goal_ob()
        last_action_ob = self.last_action

        ob = scan_ob + goal_ob + last_action_ob
        ob = np.array(ob)

        rospy.logdebug("Scan Observations==>" + str(scan_ob))
        rospy.logdebug("Goal Observations(distance, angle)==>" + str(goal_ob))
        rospy.logdebug("Last Action Observations==>" + str(last_action_ob))

        return ob

