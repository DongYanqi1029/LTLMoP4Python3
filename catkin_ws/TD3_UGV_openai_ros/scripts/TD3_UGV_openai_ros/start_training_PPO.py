#!/usr/bin/env python

import PPO
import gym
import numpy as np
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import time

if __name__ == '__main__':
    # if gpu is to be used

    rospy.init_node('my_vehicle_PPO', anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/myrobot/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('TD3_UGV_openai_ros')
    
    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    actor_path = rospy.get_param("/myrobot/actor_path")
    critic_path = rospy.get_param("/myrobot/critic_path")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = env.action_space.high

    total_timesteps = rospy.get_param("/myrobot/total_timesteps")
    timesteps_per_batch = rospy.get_param("/myrobot/timesteps_per_batch")
    max_timesteps_per_episode = rospy.get_param("/myrobot/max_timesteps_per_episode")
    n_updates_per_iteration = rospy.get_param("/myrobot/n_updates_per_iteration")
    stddev = rospy.get_param("/myrobot/stddev")
    GAMMA = rospy.get_param("/myrobot/GAMMA")
    CLIP = rospy.get_param("/myrobot/CLIP")
    ACTOR_LEARNING_RATE = rospy.get_param("/myrobot/ACTOR_LEARNING_RATE")
    CRITIC_LEARNING_RATE = rospy.get_param("/myrobot/CRITIC_LEARNING_RATE")

    policy = PPO.PPO(env, state_dim, action_dim, max_action, total_timesteps, timesteps_per_batch, max_timesteps_per_episode, n_updates_per_iteration, stddev, GAMMA, CLIP, ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, actor_path, critic_path)

    start_time = time.time()
    policy.learn()
    end_time = time.time()

    m, s = divmod(int(end_time - start_time), 60)
    h, m = divmod(m, 60)
    rospy.logerr("##### Time: %d:%02d:%02d" % (h, m, s))

    env.close()
    rospy.logerr("############Finished")