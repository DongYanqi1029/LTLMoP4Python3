#!/usr/bin/env python
"""
===================================================================
RRTController.py - Rapidly-Exploring Random Trees Motion Controller
===================================================================

Uses Rapidly-exploring Random Tree Algorithm to generate paths given the starting position and the goal point.
"""

from numpy import *
# import handlers.ROS.__is_inside.is_inside as is_inside
from handlers.share.MotionControl.__is_inside import is_inside
from lib.handlers.ROS.__DRLControllerHelper import Policy, Observation
import math
import sys,os
from scipy.linalg import norm
from numpy import zeros
import time, sys, os
import scipy as Sci
import scipy.linalg
import Polygon, Polygon.IO
import Polygon.Utils as PolyUtils
import Polygon.Shapes as PolyShapes
from math import sqrt, fabs , pi
import random
import _thread as thread


import lib.handlers.handlerTemplates as handlerTemplates

class DRLControllerHandler(handlerTemplates.MotionControlHandler):
    def __init__(self, executor, shared_data, model_name='TD3_MyRobotWorld-v0_actor', model_path='/home/dongyanqi/catkin_ws/src/TD3_UGV_openai_ros/models'):
        """
        DRL policy dependent motion planning controller

        model_name (string): NN model name
        model_path (string): NN model path
        """

        # Get references to handlers we'll need to communicate with
        self.drive_handler = executor.hsub.getHandlerInstanceByType(handlerTemplates.DriveHandler)
        self.pose_handler = executor.hsub.getHandlerInstanceByType(handlerTemplates.PoseHandler)
        self.init_handler = executor.hsub.getHandlerInstanceByType(handlerTemplates.InitHandler)

        # Get information about regions
        self.proj              = executor.proj
        self.coordmap_map2lab  = executor.hsub.coordmap_map2lab
        self.coordmap_lab2map  = executor.hsub.coordmap_lab2map
        self.last_warning      = 0
        self.previous_next_reg = None
        self.map = {}
        self.radius = self.init_handler.robotPhysicalWidth/2

        # Generate polygon for regions in the map
        for region in self.proj.rfi.regions:
            self.map[region.name] = self.createRegionPolygon(region)
            for n in range(len(region.holeList)): # no of holes
                self.map[region.name] -= self.createRegionPolygon(region, n)

        # Generate the boundary polygon
        # for regionName, regionPoly in self.map.items():
        #     self.all += regionPoly

        # Set DRL Agent
        self.model_path = model_path
        self.model_name = model_name

        self.policy = Policy(self.model_name, self.model_path)
        self.state_dim = self.policy.state_dim
        self.scan_ob_dim = self.policy.scan_ob_dim

        self.obs = Observation(self.scan_ob_dim, None, None, None)
        self.linear_vel = 0.1

        # print region info
        self.system_print = True


    def gotoRegion(self, current_reg, next_reg, last=False):
        """
        If ``last`` is True, we will move to the center of the destination region.
        Returns ``True`` if we've reached the destination region.
        """

        if current_reg == next_reg and not last:
            # No need to move!
            self.drive_handler.setVelocity(0, 0)  # So let's stop
            return True

        # Find our current configuration
        pose = self.pose_handler.getPose()
        self.current_pose = list(pose)  # list(x, y, theta, z)

        # Check if Vicon has cut out
        # TODO: this should probably go in posehandler?
        # if math.isnan(pose[2]):
        #     print("WARNING: No Vicon data! Pausing.")
        #     self.drive_handler.setVelocity(0, 0)  # So let's stop
        #     time.sleep(1)
        #     return False

        ### This part will be run when the robot goes to a new region, otherwise, the original tree will be used.
        if not self.previous_next_reg == next_reg:
            # Entered a new region. New tree should be formed.
            self.nextRegionPoly = self.map[self.proj.rfi.regions[next_reg].name]
            self.currentRegionPoly = self.map[self.proj.rfi.regions[current_reg].name]
            self.target_point = self.nextRegionPoly.center()  # tuple (x, y)

            if self.system_print is True:
                print("next Region is " + str(self.proj.rfi.regions[next_reg].name))
                print("Current Region is " + str(self.proj.rfi.regions[current_reg].name))

        Velocity = self.getVelocity(self.current_pose, self.target_point)

        self.previous_next_reg = next_reg

        # Pass this desired velocity on to the drive handler
        self.drive_handler.setVelocity(Velocity[0, 0], Velocity[1, 0])

        RobotPoly = Polygon.Shapes.Circle(self.radius, (pose[0], pose[1]))

        # check if robot is inside the current region
        departed = not self.currentRegionPoly.overlaps(RobotPoly)
        arrived = self.nextRegionPoly.covers(RobotPoly)

        if departed and (not arrived) and (time.time() - self.last_warning) > 0.5:
            # Figure out what region we think we stumbled into
            for r in self.proj.rfi.regions:
                pointArray = [self.coordmap_map2lab(x) for x in r.getPoints()]
                vertices = mat(pointArray).T

                if is_inside([pose[0], pose[1]], vertices):
                    print("I think I'm in " + r.name)
                    print(pose)
                    break
            self.last_warning = time.time()

        # print "arrived:"+str(arrived)
        return arrived

    def createRegionPolygon(self, region, hole=None):
        """
        This function takes in the region points and make it a Polygon.
        """
        if hole == None:
            pointArray = [x for x in region.getPoints()]
        else:
            pointArray = [x for x in region.getPoints(hole_id = hole)]
        pointArray = map(self.coordmap_map2lab, pointArray)
        regionPoints = [(pt[0],pt[1]) for pt in pointArray]
        formedPolygon = Polygon.Polygon(regionPoints)
        return formedPolygon

    def getVelocity(self, pose, target):
        """
        This function calculates the velocity for the robot with DRL Policy.
        The inputs are (given in order):
            pose           = the current pose of the robot
            target        = the x-y position of the target point
        """

        # print("Current pose: " + str(pose))
        # print("Current target: " + str(target))

        # DRL
        self.obs.pose = pose
        self.obs.target = list(target)
        self.obs.last_action = list(self.policy.last_action)

        state = self.obs.get_state()
        action = self.policy.get_action(state)

        Vel = zeros([2,1])
        Vel[0, 0] = self.linear_vel
        Vel[1, 0] = action[0]

        return Vel