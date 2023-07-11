#!/usr/bin/env python

import numpy as np
import torch
import gym
import argparse
import os
import rospy
import rospkg

import utils
import TD3
# import OurDDPG
# import DDPG
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
# evaluate与train使用同一个env
def eval_policy(policy, env, eval_episodes=5, max_episode_timesteps=100):
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, _ = env.reset()
		done = False
		step = 1
		while (not done) and step <= max_episode_steps:
			action = policy.select_action(state)
			state, reward, done, _, _ = env.step(action)
			avg_reward += reward
			step += 1

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":

	rospy.init_node("MyRobot_TD3", anonymous=True, log_level=rospy.WARN)

	# Init OpenAI_ROS ENV
	task_and_robot_environment_name = rospy.get_param('/myrobot/task_and_robot_environment_name')
	env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
	training_path = rospy.get_param('/myrobot/training_path')

	rospy.loginfo("OpenAI environment done")
	rospy.loginfo("Starting Learning")

	# Set the logging system
	rospack = rospkg.RosPack()
	pkg_path = rospack.get_path('TD3_UGV_openai_ros')

	# Load parameters from yaml config file
	# model_path = rospy.get_param('/myrobot/model_path')

	policy_name = rospy.get_param('/myrobot/policy_name')
	start_timesteps = rospy.get_param('/myrobot/start_timesteps') # Time steps initial RANDOM policy is used
	eval_freq =  rospy.get_param('/myrobot/eval_freq') # How often (time steps) we evaluate
	max_timesteps = rospy.get_param('/myrobot/max_timesteps') # Max time steps to run environment
	max_episode_steps = rospy.get_param('/myrobot/max_episode_steps') # Max time steps per episode
	expl_noise = rospy.get_param('/myrobot/expl_noise') # Std of Gaussian exploration noise
	batch_size = rospy.get_param('/myrobot/batch_size') # Batch size for both actor and critic
	discount = rospy.get_param('/myrobot/reward_discount_factor')
	tau = rospy.get_param('/myrobot/tau')
	policy_noise = rospy.get_param('/myrobot/policy_noise')
	noise_clip = rospy.get_param('/myrobot/noise_clip')
	policy_freq = rospy.get_param('/myrobot/policy_freq')
	save_model = rospy.get_param('/myrobot/save_model')
	load_model = rospy.get_param('/myrobot/load_model')
	
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	# parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
	# parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	# parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	# parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	# parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	# parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	# parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	# parser.add_argument("--discount", default=0.99)                 # Discount factor
	# parser.add_argument("--tau", default=0.005)                     # Target network update rate
	# parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	# parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	# parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	# parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# args = parser.parse_args()

	file_name = f"{policy_name}_{task_and_robot_environment_name}"
	print("---------------------------------------")
	print(f"Policy: {policy_name}, Env: {task_and_robot_environment_name}")
	print("---------------------------------------")

	if not os.path.exists(training_path + "/results"):
		os.makedirs(training_path + "/results")

	if save_model and not os.path.exists(training_path + "/models"):
		os.makedirs(training_path + "/models")

	# Set seeds
	# env.seed(args.seed)
	# env.action_space.seed(args.seed)
	# torch.manual_seed(args.seed)
	# np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = env.action_space.high

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": discount,
		"tau": tau,
	}

	# Initialize policy
	policy = None
	if policy_name == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = policy_noise * max_action
		kwargs["noise_clip"] = noise_clip * max_action
		kwargs["policy_freq"] = policy_freq
		policy = TD3.TD3(**kwargs)
	# elif args.policy == "OurDDPG":
	# 	policy = OurDDPG.DDPG(**kwargs)
	# elif args.policy == "DDPG":
	# 	policy = DDPG.DDPG(**kwargs)

	if load_model != "":
		policy_file = file_name if load_model == "default" else load_model
		policy.load(training_path + f"/models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, env)]

	state, _ = env.reset()
	done = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		rospy.logwarn("Step " + str(episode_timesteps) + " : ")

		# Perform action
		next_state, reward, done, truncted, _ = env.step(action) 

		done_bool = float(done) if episode_timesteps < max_episode_steps else 1

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, float(done))

		rospy.logwarn("Get state => " + str(state))

		state = next_state
		episode_reward += reward

		rospy.logwarn("Set action => " + str(action))
		rospy.logwarn("Get next state => " + str(next_state))
		rospy.logwarn("Get reward => " + str(reward))
		rospy.logwarn("Total reward => " + str(episode_reward))

		# Train agent after collecting sufficient data
		if t >= start_timesteps:
			policy.train(replay_buffer, batch_size)

		if done_bool: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, _ = env.reset()
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % eval_freq == 0:
			evaluations.append(eval_policy(policy, env))
			np.save(training_path + f"/results/{file_name}", evaluations)
			if save_model: policy.save(training_path + f"/models/{file_name}")