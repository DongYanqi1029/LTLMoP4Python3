import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import myrobot_env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os

from gazebo_msgs.srv import *
from transforms3d.euler import quat2euler
import cmath
# from enum import IntEnum, unique

# class Action():
#     def __init__(self, linear_vel, angular_vel):
#         self.linear_vel = linear_vel
#         self.angular_vel = angular_vel

#     def set_linear_vel(self, linear_vel):
#         self.linear_vel = linear_vel

#     def get_linear_vel(self):
#         return self.linear_vel

#     def set_angular_vel(self, linear_vel):
#         self.angular_vel = angular_vel

#     def get_angular_vel(self):
#         return self.angular_vel 

# @unique
# class EPISODE_DONE(IntEnum):
#     NOT_DONE = 0
#     REACH = 1
#     CRASH = 2

class MyRobotWorldEnv(myrobot_env.MyRobotEnv):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/myrobot/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/myrobot/config",
                               yaml_file_name="myrobot_world.yaml")

        # Start task gazebo environment
        task_ros_package = rospy.get_param('/myrobot/task_ros_package')
        task_launch_file = rospy.get_param('/myrobot/task_launch_file')

        ROSLauncher(rospackage_name=task_ros_package,
                    launch_file_name=task_launch_file,
                    ros_ws_abspath=ros_ws_abspath)

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(MyRobotWorldEnv, self).__init__(ros_ws_abspath)

        #number_observations = rospy.get_param('/turtlebot3/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """

        # Actions and Observations
        self.linear_speed = rospy.get_param('/myrobot/linear_speed')
        self.max_linear_speed = rospy.get_param('/myrobot/max_linear_speed')
        self.max_angular_speed = rospy.get_param('/myrobot/max_angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/myrobot/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/myrobot/init_linear_turn_speed')
        self.turn_threshold = rospy.get_param('/myrobot/turn_threshold')
        self.forward_threshold = rospy.get_param('/myrobot/forward_threshold')

        self.scan_ob_dim = rospy.get_param('/myrobot/scan_ob_dim')
        self.goal_ob_dim = rospy.get_param('/myrobot/goal_ob_dim')
        self.last_action_ob_dim = rospy.get_param('/myrobot/last_action_ob_dim')
        self.goal_range = rospy.get_param('/myrobot/goal_range')
        self.min_range = rospy.get_param('/myrobot/min_range')
        self.safe_distance = rospy.get_param('/myrobot/safe_distance')
        self.max_laser_value = rospy.get_param('/myrobot/max_laser_value')
        self.min_laser_value = rospy.get_param('/myrobot/min_laser_value')
        # self.max_linear_aceleration = rospy.get_param('/turtlebot3/max_linear_aceleration')

         # Get robot model name
        self.model_name = rospy.get_param('/myrobot/model_name')

        # Only variable needed to be set here
        action_high = numpy.array([self.max_angular_speed])
        action_low = numpy.array([-self.max_angular_speed])
        self.action_space = spaces.Box(action_low, action_high, dtype=numpy.float64)
        self.last_action = numpy.array([0.0])

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        
        # scan_ob range
        scan_high = numpy.full((self.scan_ob_dim), self.max_laser_value)
        scan_low = numpy.full((self.scan_ob_dim), self.min_laser_value)

        # goal_ob and last_action_ob range
        pi = 3.14
        high = numpy.array([numpy.inf, pi, self.max_angular_speed])
        low = numpy.array([-numpy.inf, -pi, -self.max_angular_speed])

        ob_high = numpy.append(scan_high, high)
        ob_low = numpy.append(scan_low, low)

        # We only use two integers
        self.observation_space = spaces.Box(ob_low, ob_high, dtype=numpy.float64)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Goal realted
        self.robot_radius = rospy.get_param('/myrobot/robot_radius')
        self.goal_x = rospy.get_param('/myrobot/goal_point_x')
        self.goal_y = rospy.get_param('/myrobot/goal_point_y')

        # Rewards
        self.reward_baseline = rospy.get_param('/myrobot/reward_baseline')
        self.reward_scale_factor = rospy.get_param('/myrobot/reward_scale_factor')
        self.reach_target_reward = rospy.get_param('/myrobot/reach_target_reward')
        self.crash_penalty = rospy.get_param('/myrobot/crash_penalty')
        self.forwards_reward = rospy.get_param('/myrobot/forwards_reward')
        self.backwards_reward = rospy.get_param('/myrobot/backwards_reward')
        self.straight_reward = rospy.get_param('/myrobot/straight_reward')
        self.turn_reward = rospy.get_param('/myrobot/turn_reward') 

        self.cumulated_steps = 0


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

    
    def get_robot_pose(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            request = GetModelStateRequest()
            # print(self.model_name)
            request.model_name = self.model_name
            model_state = gms(request)
            #Cartesian Pose
            self.robot_pos_x = model_state.pose.position.x
            self.robot_pos_y = model_state.pose.position.y
            self.robot_pos_z = model_state.pose.position.z
            #Quaternion Orientation
            or_x = model_state.pose.orientation.x
            or_y = model_state.pose.orientation.y
            or_z = model_state.pose.orientation.z
            or_w = model_state.pose.orientation.w
            angles = quat2euler([or_w, or_x, or_y, or_z])
            self.robot_or_alpha = angles[0]
            self.robot_or_beta = angles[1]
            self.robot_or_theta = angles[2] # [-pi, pi]
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    
    def get_goal_ob(self):
        self.get_robot_pose()
        # 相对x, y坐标
        relative_x = self.goal_x - self.robot_pos_x
        relative_y = self.goal_y - self.robot_pos_y
        coor = complex(relative_x, relative_y)

        # 极坐标
        # 在机器人朝向左侧为正, 右侧为负
        distance, angle = cmath.polar(coor) # [-pi, pi]
        angle = angle - self.robot_or_theta
        pi = 3.14
        if (angle < -pi): # 相当于在左侧
            angle += 2*pi
        if (angle > pi): # 相当于在右侧
            angle -= 2*pi

        goal_ob = [distance, angle]

        return goal_ob


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>" + str(action[0]))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        # linear_vel = action[0]
        angular_vel = action[0]
        self.last_action = action

        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(self.linear_speed, angular_vel, epsilon=0.05, update_rate=10)

        rospy.logdebug("END Set Action ==>" + str(angular_vel))

    def get_last_action_ob(self):
        last_action_ob = [self.last_action[0]]
        return last_action_ob

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()

        scan_ob = self.get_observation_from_scan(laser_scan, self.scan_ob_dim)
        goal_ob = self.get_goal_ob()
        last_action_ob = self.get_last_action_ob()

        ob = scan_ob + goal_ob + last_action_ob
        ob = numpy.array(ob)

        rospy.logdebug("Scan Observations==>"+str(scan_ob))
        rospy.logdebug("Goal Observations(distance, angle)==>"+str(goal_ob))
        rospy.logdebug("Last Action Observations==>"+str(last_action_ob))
        rospy.logdebug("END Get Observation ==>")
        return ob


    def _is_done(self, observations):

        # Now we check if it has crashed based on the observation
        # min_scan_data = min(observations[:self.scan_ob_dim])
        distance = observations[self.scan_ob_dim]

        if self.is_crash():
            self._episode_done = True
            rospy.logerr("Robot is Too Close to wall")
        elif distance < self.goal_range:
            self._episode_done = True
            rospy.logwarn("Robot REACH target point.")
        else:
            self._episode_done = False
            rospy.logwarn("Robot is NOT close to a wall.")

        return self._episode_done

    def _compute_reward(self, observations, done):

        reward = 0

        if not done:
            distance_to_goal = observations[self.scan_ob_dim]
            # lin_vel = observations[self.scan_ob_dim + 2]
            # ang_vel = observations[self.scan_ob_dim + 3]
            # if lin_vel >= self.forward_threshold:
            #     reward = self.forwards_reward
            # else:
            #     reward = self.backwards_reward

            
            # if abs(ang_vel) < self.turn_threshold:
            #     reward += self.straight_reward
            # else:
            #     reward += self.turn_reward5

            # if distance_to_goal < (self.robot_radius + self.safe_distance):
            #     reward = 1 - (distance_to_goal/(self.robot_radius + self.safe_distance))
            # else:
            #     reward = self.reward_scale_factor * (self.goal_range - distance_to_goal)
            reward = self.reward_baseline + self.reward_scale_factor * (self.goal_range - distance_to_goal)
        else:
            # min_scan_data = min(observations[:self.scan_ob_dim])
            distance = observations[self.scan_ob_dim]
            if self.is_crash(): # crash
                reward = self.crash_penalty
            elif distance < self.goal_range: # reach
                reward = self.reach_target_reward

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward


    # Internal TaskEnv Methods

    def get_observation_from_scan(self, data, scan_ob_dim):
        # 获取全方位360度数据

        scan_ob = []
        laser_data = list(data.ranges)
        # laser_data_left = laser_data[300:]
        # laser_data_right = laser_data[0:61]
        # laser_data = laser_data_left + laser_data_right
        mod = (len(laser_data)//scan_ob_dim)

        for i, item in enumerate(laser_data):
            if (i%mod==0):
                # laser_data单位:m
                # distance = item * 100
                distance = item
                if distance == float ('Inf') or numpy.isinf(distance):
                    scan_ob.append(self.max_laser_value)
                elif numpy.isnan(distance):
                    scan_ob.append(self.min_laser_value)
                else:
                    scan_ob.append(distance)
        
        return scan_ob

    def get_vector_magnitude(self, vector):
        """
        It calculated the magnitude of the Vector3 given.
        This is usefull for reading imu accelerations and knowing if there has been
        a crash
        :return:
        """
        contact_force_np = numpy.array((vector.x, vector.y, vector.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude

    def is_crash(self):
        data = list(self.get_laser_scan().ranges)
        dist = []
        for distance in data:
                # data单位:m
                if distance == float ('Inf') or numpy.isinf(distance):
                    dist.append(self.max_laser_value)
                elif numpy.isnan(distance):
                    dist.append(self.min_laser_value)
                else:
                    dist.append(distance)
        min_scan_data = min(data)
        if min_scan_data < self.min_range:
            return True
        else:
            return False



