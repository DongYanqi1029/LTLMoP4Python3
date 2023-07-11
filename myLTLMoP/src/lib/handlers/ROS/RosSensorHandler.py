#!/usr/bin/env python
"""
====================================================
rosSensor.py - Sensor handler for the ROS interface
====================================================
"""

import roslib
import rospy
from sensor_msgs.msg import LaserScan

import lib.handlers.handlerTemplates as handlerTemplates

class RosSensorHandler(handlerTemplates.SensorHandler):
	def __init__(self, executor, shared_data):
		"""
		ROS Sensor Handler
		"""
		self.rosInitHandler = shared_data['ROS_INIT_HANDLER']

	###################################
	### Available sensor functions: ###
	###################################

	def _subscriber(self, initial=False):
		"""
		This is a template for a subscriber type sensor interface
		"""
		self.returnVal=False

		def processData(data, sensor):
			# This gets called from the Subscriber
			# Do some processing here
			# Set the returnVal to either True or False based on the data
			# Data is a message, its properties are what you want to evaluate
			# sensor is the sensorHandler (self)
			sensor.returnVal = False

		if initial:
			# This is the topic you want to listen to
			self.topic = ''
			# This is the message type that the topic broadcasts
			# It is also the data type that gets passed to processData
			self.messageType = ''
		else:
			# There is already an initiated node for LTLMoP
			# Otherwise we would create one
			# This creates a subscriber for the given topic
			# Pass the topic, messageType, the function to call,
			# and self (for global variable access)
			rospy.Subscriber(self.topic, eval(self.messageType), processData, self)
			# You must return boolean
			return self.returnVal

	def _service(self, initial=False):
		"""
		This is a template for a service type sensor interface
		"""
		def handleData(data):
			# Do some processing to the data here
			# Return a boolean based on the data
			return False
		if initial:
			# This is the name of the service you want to create
			self.serviceName=''
			# This is the message type for the service
			self.messageType=''
		else:
		# There is already an initiated node for LTLMoP
			# This creates the service
			response = rospy.Service(self.serviceName, self.messageType, handleData)
			return response

	def avgScanData(self, initial=False):
		"""
		This is a sample sensor interface printing scan data
		"""

		self.returnVal = False

		def processData(data, sensor):
			avgRange = 0
			try:
				avgRange = sum(list(data.ranges))/len(list(data.ranges))
			except:
				print('No Range Data')
			# print(avgRange)
			sensor.returnVal = False

		if initial:
			self.topic = '/scan'
			self.messageType = LaserScan
		else:
			try:
				rospy.Subscriber(self.topic, self.messageType, processData, self)
			except:
				print("Subscriber Failed")

			return self.returnVal
