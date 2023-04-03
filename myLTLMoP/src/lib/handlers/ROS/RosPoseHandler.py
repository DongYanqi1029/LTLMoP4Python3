#!/usr/bin/env python
import roslib
# roslib.load_manifest('gazebo')

import rospy, math
from gazebo_msgs.srv import *
from numpy import *
from std_msgs.msg import String
# from tf.transformations import euler_from_quaternion
from transforms3d.euler import quat2euler

"""
=======================================
rosPose.py - ROS Interface Pose Handler
=======================================
"""

import lib.handlers.handlerTemplates as handlerTemplates

class RosPoseHandler(handlerTemplates.PoseHandler):
	def __init__(self, executor, shared_data, modelName="turtlebot3_burger"):
		"""
		Pose Handler for ROS and gazebo.

		modelName (str): The model name of the robot in gazebo to get the pose information from (default="pr2")
		"""

		#GetModelState expects the arguments model_name and relative_entity_name
		#In this case it is pr2 and world respectively but can be changed for different robots and environments
		self.model_name = modelName
		self.relative_entity_name = 'world' #implies the gazebo global coordinates
		self.last_pose = None

		self.shared_data = shared_data['ROS_INIT_HANDLER']

	def getPose(self, cached=False):
		if (not cached) or self.last_pose is None:
			#Ros service call to get model state
			#This returns a GetModelStateResponse, which contains data on pose
			rospy.wait_for_service('/gazebo/get_model_state')
			try:
				gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
				request = GetModelStateRequest()
				# print(self.model_name)
				request.model_name = self.model_name
				model_state = gms(request)
				#Cartesian Pose
				self.pos_x = model_state.pose.position.x
				self.pos_y = model_state.pose.position.y
				self.pos_z = model_state.pose.position.z
				# print((self.pos_x, self.pos_y, self.pos_z))
				#Quaternion Orientation
				self.or_x = model_state.pose.orientation.x
				self.or_y = model_state.pose.orientation.y
				self.or_z = model_state.pose.orientation.z
				self.or_w = model_state.pose.orientation.w
			except rospy.ServiceException as e:
				print("Service call failed: %s"%e)
			#  Use the tf module transforming quaternions to euler
			try:
				angles = quat2euler([self.or_w, self.or_x, self.or_y, self.or_z])
				self.theta = angles[2]
				shared = self.shared_data
				#The following accounts for the maps offset in gazebo for
				#initial region placement
				self.last_pose = array([self.pos_x+shared.offset[0], self.pos_y+shared.offset[1], self.theta, self.pos_z])
			except Exception as e:
				print('Pose Broke: ' + str(e))
		# print(self.last_pose)
		return self.last_pose
