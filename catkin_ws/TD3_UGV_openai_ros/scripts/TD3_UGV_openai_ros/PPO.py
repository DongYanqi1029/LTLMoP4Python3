import torch
import rospy
from torch import nn
import numpy as np
import networks
from torch.distributions import Normal
from torch.optim import Adam
import sys
import matplotlib
import matplotlib.pyplot as plt
import os


class PPO:
    def __init__(self, env, state_dim, action_dim, max_action, total_timesteps=24000, timesteps_per_batch=4800, max_timesteps_per_episode=1600, n_updates_per_iteration=5, stddev = 0.5, GAMMA=0.95, clip=0.2, actor_lr=0.005, critic_lr=0.005, actor_path=None, critic_path=None):
        self._init_hyperparameters(total_timesteps, timesteps_per_batch, max_timesteps_per_episode, n_updates_per_iteration, stddev, GAMMA, clip, actor_lr, critic_lr, actor_path, critic_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract environment information
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # ALG STEP 1
        # Initialize actor and critic networks
        # self.actor = network.LaserNetActor(self.obs_dim).to(self.device)
        # self.critic = network.LaserNetCritic(self.obs_dim).to(self.device)
        self.actor = networks.PPOActor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.critic = networks.PPOCritic(self.state_dim).to(self.device)

        if os.path.exists(self.actor_path):
            self.actor.load_state_dict(torch.load(self.actor_path))
        if os.path.exists(self.critic_path):
            self.critic.load_state_dict(torch.load(self.critic_path))
            
        self.actor.train()
        self.critic.train()

        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

        self.episodes_reward = []

    def _init_hyperparameters(self, total_timesteps, timesteps_per_batch, max_timesteps_per_episode, n_updates_per_iteration, stddev, GAMMA, clip, actor_lr, critic_lr, actor_path, critic_path):
        # Default values for hyperparameters, will need to change later.
        self.total_timesteps = total_timesteps
        self.timesteps_per_batch = timesteps_per_batch            
        self.max_timesteps_per_episode = max_timesteps_per_episode  
        self.n_updates_per_iteration = n_updates_per_iteration  
        self.stddev = stddev  
        self.GAMMA = GAMMA
        self.clip = clip 
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_path = actor_path
        self.critic_path = critic_path

    def rollout(self):
        # 获取一批次的交互数据，多个episode
        # Batch data
        batch_obs = []             # batch observations
        batch_actions = []         # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rewards = []         # batch rewards
        batch_rewards_to_go = []   # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        # Number of timesteps run so far this batch
        t = 0 
        i_episode = 1
        while t < self.timesteps_per_batch:
            rospy.logwarn("##### START EPISODE => " + str(i_episode))
            # Rewards this episode
            ep_rewards = []
            obs, _ = self.env.reset()
            state = torch.tensor(np.array([obs]), dtype=torch.float).to(self.device)
            # rospy.logwarn("State size => " + str(state.size()))
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                rospy.logwarn("##### START STEP => " + str(ep_t + 1))
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                rospy.logwarn("State => " + str(state))
                batch_obs.append(state)
                action, log_prob = self.get_action(state)
                obs, reward, done, _, _= self.env.step(np.array(action[0].cpu()))
                state = torch.tensor(np.array([obs]), dtype=torch.float).to(self.device)
                rospy.logwarn("Reward that action give => " + str(reward))
            
                # Collect reward, action, and log prob
                ep_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)

                rospy.logwarn("##### END STEP => " + str(ep_t + 1))

                if done or t >= self.timesteps_per_batch:
                    break

            # Collect episodic length and rewards
            rospy.logwarn("##### END EPISODE => " + str(i_episode))
            rospy.logwarn("##### EPISODE REWARD => " + str(sum(ep_rewards)))

            i_episode += 1
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rewards.append(ep_rewards) 

            self.episodes_reward.append(sum(ep_rewards))
            # self.plot_durations(self.episodes_reward)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.cat(batch_obs, 0)
        batch_actions = torch.cat(batch_actions, 0)
        batch_log_probs = torch.cat(batch_log_probs, 0)
        # ALG STEP #4
        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)

        # Return the batch data
        return batch_obs, batch_actions, batch_log_probs, batch_rewards_to_go, batch_lens

    def get_action(self, state):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(state)
        angular_speed_mean = self.actor(state)
        rospy.logwarn("Angular speed mean => " + str(angular_speed_mean))

        # Create our Multivariate Normal Distribution
        dist = Normal(angular_speed_mean.squeeze(1), torch.tensor([self.stddev]).to(self.device))
        rospy.logwarn("Angular speed distribution => " + str(dist))

        # Sample an action from the distribution and get its log prob
        action = dist.sample().unsqueeze(0)
        rospy.logwarn("Action sampled => " + str(action))
        log_prob = dist.log_prob(action)
        rospy.logwarn("Prob density for sample => " + str(log_prob.exp()))
        
        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach(), log_prob.detach()

    def compute_rewards_to_go(self, batch_rewards):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0 # The discounted reward so far
            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.GAMMA
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device).unsqueeze(1)
        return batch_rtgs

    def evaluate(self, batch_obs, batch_actions):
        # rospy.logwarn("Evaluate batch_obs => " + str(batch_obs))
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs)
        # rospy.logwarn("Evaluate V => " + str(V))

        # Calculate the log probabilities of batch actions using most 
        # recent actor network.
        # This segment of code is similar to that in get_action()
        angular_speed_means = self.actor(batch_obs)
        rospy.logdebug("Evaluate angular speed mean => " + str(angular_speed_means))

        dist = Normal(angular_speed_means.squeeze(1), torch.tensor([self.stddev]).to(self.device))
        log_probs = dist.log_prob(batch_actions.squeeze(1)).unsqueeze(1)
        rospy.logdebug("Evaluate angular speed distribution => " + str(dist))
        rospy.logdebug("Evaluate actions => " + str(batch_actions))
        rospy.logdebug("Evaluate actions probs density => " + str(log_probs.exp()))

        # Return predicted values V and log probs log_probs
        return V, log_probs

    def learn(self):
        rospy.logerr("##### Start Learning #####")
        t_so_far = 0 # Timesteps simulated so far
        i_batch = 1
        while t_so_far < self.total_timesteps:              # ALG STEP 2
            # Increment t_so_far somewhere below
            # ALG STEP 3
            rospy.logerr("##### Learning batch => " + str(i_batch))
            batch_obs, batch_actions, batch_log_probs, batch_rewards_to_go, batch_lens = self.rollout()
            rospy.logwarn("Rollout observations => " + str(batch_obs))
            rospy.logwarn("Rollout actions => " + str(batch_actions))
            rospy.logwarn("Rollout log_probs density=> " + str(batch_log_probs.exp()))
            rospy.logwarn("Rollout batch_rtgs => " + str(batch_rewards_to_go))
            rospy.logwarn("Rollout batch_lens => " + str(len(batch_lens)))
            rospy.logerr("##### Batch sample finished #####")

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_actions)

            # ALG STEP 5
            # Calculate advantage
            A_k = (batch_rewards_to_go - V.detach()).squeeze()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            rospy.logwarn("Advantage functions for batch " + str(i_batch) + " => " + str(A_k))
            rospy.logwarn("Value function estimation for batch " + str(i_batch) + " => " + str(V))

            for i_update in range(self.n_updates_per_iteration):
                rospy.logwarn("Update iteration => " + str(i_update + 1))
                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_actions)

                # Calculate ratios
                ratios = torch.exp((curr_log_probs - batch_log_probs).squeeze())
                rospy.logdebug("Value function for batch " + str(i_batch) + ", update " + str(i_update + 1) + " => " + str(V))
                rospy.logdebug("curr_log_probs for batch " + str(i_batch) + ", update " + str(i_update + 1) + " => " + str(curr_log_probs))
                rospy.logwarn("Ratios for batch " + str(i_batch) + ", update " + str(i_update + 1) + " => " + str(ratios))

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                rospy.logwarn("Surr1 for batch " + str(i_batch) + ", update " + str(i_update + 1) + " => " + str(surr1))
                rospy.logwarn("Surr2 for batch " + str(i_batch) + ", update " + str(i_update + 1) + " => " + str(surr2))

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V.squeeze(), batch_rewards_to_go.squeeze())
                rospy.logerr("Actor loss for batch " + str(i_batch) + ", update " + str(i_update + 1) + " => "  + str(actor_loss))
                rospy.logerr("Critic loss for batch "+ str(i_batch) + ", update " + str(i_update + 1) + " => "  + str(critic_loss))

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network    
                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()

                self.save_model()

            # Calculate how many timesteps we collected this batch  
            t_so_far += np.sum(batch_lens)
            i_batch += 1

            # sys.exit()


        rospy.logerr("##### Learning Finished #####")




    def save_model(self):
        # torch.save({'state_dict': self.actor.state_dict()}, self.actor_path)
        torch.save(self.actor.state_dict(), self.actor_path, _use_new_zipfile_serialization=False)
        torch.save(self.critic.state_dict(), self.critic_path, _use_new_zipfile_serialization=False)

    def plot_durations(self, reward_data):
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        plt.ion()

        plt.figure(1)
        # plt.clf消除之前的plot
        plt.clf()
        total_reward = torch.tensor(reward_data, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.plot(total_reward.numpy())
        # Take 100 episode averages and plot them too
        # 绘制最近100个episode的平均reward
        if len(total_reward) >= 100:
            means = total_reward.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99)-200, means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        fig = plt.gcf()
        fig.savefig("/home/diy/catkin_ws/src/my_vehicle_learning/training_results/training.png")
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

class Testing:
    def __init__(self, policy_net):
        self.policy_net = policy_net
    def selectAction(self, state):
        with torch.no_grad():
            return self.policy_net(state)