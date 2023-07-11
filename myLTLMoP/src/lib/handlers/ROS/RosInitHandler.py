
#!/usr/bin/env python
"""
=================================================
rosSim.py - ROS/Gazebo Initialization Handler
=================================================
"""
import math
import roslib
# roslib.load_manifest('gazebo')
import sys, subprocess, os, time, os, shutil, rospy

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

import re, Polygon, Polygon.IO
import lib.regions as regions
#import execute
from numpy import *
#from gazebo.srv import *
from gazebo_msgs.srv import *
from gazebo_ros import gazebo_interface
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
import fileinput

import lib.handlers.handlerTemplates as handlerTemplates
import logging

from Cheetah.Template import Template
from lib.globalConfig import get_ltlmop_root

class RosInitHandler(handlerTemplates.InitHandler):
    def __init__(self, executor, init_region, worldFile='ltlmop_map.world', robotPixelWidth=40, robotPhysicalWidth=0.5, robotPackage="simulator_gazebo", robotLaunchFile="turtlebot3_burger.launch", modelName = "turtlebot3_burger"):
        """
        Initialization handler for ROS and gazebo

        init_region (region): The initial region of the robot
        worldFile (str): The alternate world launch file to be used (default="ltlmop_map.world")
        robotPixelWidth (int): The width of the robot in pixels in ltlmop (default=200)
        robotPhysicalWidth (float): The physical width of the robot in meters (default=0.5)
        robotPackage (str): The package where the robot is located (default="pr2_gazebo")
        robotLaunchFile (str): The launch file name for the robot in the package (default="pr2.launch")
        modelName(str): Name of the robot. Choices: pr2 and quadrotor for now(default="pr2")
        """

        self.proj = executor.proj
        # Set a blank offset for moving the map
        self.offset=[0,0]
        # The package and launch file for the robot that is being used
        self.package = robotPackage
        self.launchFile = robotLaunchFile
        # The world file that is to be launched, see gazebo_worlds/worlds
        self.worldFile = worldFile
        # Map to real world scaling constant
        self.ratio = robotPhysicalWidth/robotPixelWidth
        self.robotPhysicalWidth = robotPhysicalWidth
        self.modelName = modelName
        self.coordmap_map2lab = executor.hsub.coordmap_map2lab

        # change the starting pose of the box
        self.original_regions = self.proj.loadRegionFile()
        self.region_file_name = self.proj.spec_data['SETTINGS']['RegionFile'][0].rstrip('.regions')
        self.spec_file_name = self.proj.getFilenamePrefix().split('/')[-1]
        self.root = get_ltlmop_root()
        self.tmpl_path = self.root + '/lib/handlers/ROS/templates/'


        self.worldFile = self.region_file_name + "_region_map.world"
        self.mapPic = self.region_file_name + '_region_map.png'
        self.materialFile = self.region_file_name + '_region_map.material'

        self.destination = "/home/dongyanqi/catkin_ws/src/simulator_gazebo/worlds/" + self.worldFile

        # Center the robot in the init region (not on calibration)
        if init_region == "__origin__":
            os.environ['ROBOT_INITIAL_POSE'] = "-x " + str(0) + " -y " + str(0)
        else:
            self.centerTheRobot(executor, init_region)

        # Create world file
        self.worldNamespace = {}

        f = open(self.tmpl_path + 'region_map.world.tmpl', 'r')
        worldDef = f.read()
        f.close()

        # Set world name
        worldName = self.region_file_name + '_world'
        self.worldNamespace['world_name'] = worldName

        # This creates a png copy of the regions to load into gazebo
        self.createRegionMap(executor.proj)

        # Add obstacle
        self.addObstacles()

        # Create Boundary
        self.createBoundary()

        # Create file
        f = open(self.destination, 'w+')
        world = Template(worldDef, searchList=self.worldNamespace)
        f.write(str(world))
        f.close()

        # Change robot and world file in gazebo:
        self.changeRobotAndWorld()

        # set up a publisher to publish pose
        self.pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)

        # Create a subprocess for ROS
        self.rosSubProcess(executor.proj)

        #The following is a global node for LTLMoP
        rospy.init_node('LTLMoPHandlers')

    def createRegionPolygon(self, region, hole=None):
        """
        This function takes in the region points and make it a Polygon.
        """
        if hole is None:
            pointArray = [x for x in region.getPoints()]
        else:
            pointArray = [x for x in region.getPoints(hole_id=hole)]

        pointArray = map(self.coordmap_map2lab, pointArray)
        regionPoints = [(pt[0], pt[1]) for pt in pointArray]
        formedPolygon = Polygon.Polygon(regionPoints)
        return formedPolygon

    def getSharedData(self):
        # TODO: Return a dictionary of any objects that will need to be shared with other handlers
        return {'ROS_INIT_HANDLER': self}

    def region2svg(self, proj, regionFile):
        """
        Converts region file to svg
        This is from the deprecated regions file with slight changes for
        proper calculation of the size of the regions map
        """
        fout = re.sub(r"\.region$", ".svg", regionFile)
        rfi = regions.RegionFileInterface()
        rfi.readFile(regionFile)

        polyList = []

        for region in rfi.regions:
            points = [(pt.x, -pt.y) for pt in region.getPoints()]
            poly = Polygon.Polygon(points)
            polyList.append(poly)
        try: # Normal Operation
            boundary = proj.rfiold.regions[proj.rfiold.indexOfRegionWithName("boundary")]
        except: # Calibration
            boundary = proj.rfi.regions[proj.rfi.indexOfRegionWithName("boundary")]
        width = boundary.size.width
        height = boundary.size.height

        # use boundary size for image size
        # Polygon.IO.writeSVG(fout, polyList,width=width,height=height)
        Polygon.IO.writeSVG(fout, polyList, width=height, height=width)   # works better than width=width,height=height


        return fout  # return the file name

    def createRegionMap(self, proj):
        """
        This function creates the ltlmop region map as a floor plan in the
        Gazebo Simulator.
        """
        # This block creates a copy and converts to svg
        texture_dir = '/home/dongyanqi/catkin_ws/src/simulator_gazebo/materials/textures' #potentially dangerous as pathd in ROS change with updates
        ltlmop_path = proj.getFilenamePrefix()
        regionsFile = ltlmop_path + "_copy.regions"
        shutil.copy(proj.rfi.filename, regionsFile)
        svgFile = self.region2svg(proj, regionsFile) # svg file name
        drawing = svg2rlg(svgFile)
        
        # This block converts the svg to png and applies naming conventions
        renderPM.drawToFile(drawing, ltlmop_path+"_simbg.png", fmt='PNG')
        ltlmop_map_path = ltlmop_path + "_simbg.png"
        shutil.copy(ltlmop_map_path, texture_dir)
        full_pic_path = texture_dir + "/" + proj.project_basename + "_simbg.png"
        texture_pic_path = texture_dir + "/" + self.mapPic
        shutil.copyfile(full_pic_path, texture_pic_path)

        # Create material file
        f = open(self.tmpl_path + 'region_map.material.tmpl', 'r')
        materialDef = f.read()
        f.close()

        materialNamespace = {'map_pic': self.mapPic}
        material = Template(materialDef, searchList=[materialNamespace])

        material_path = '/home/dongyanqi/catkin_ws/src/simulator_gazebo/materials/scripts/' + self.materialFile
        f = open(material_path, 'w+')
        f.write(str(material))
        f.close()

        # Change size of region map in gazebo
        from PIL import Image
        img = Image.open(texture_dir + "/" + self.mapPic)
        imgWidth = img.width
        imgHeight = img.height
        img.close()

        # This is accomplished through edits of the world file before opening
        T = [self.ratio * imgWidth, self.ratio * imgHeight]
        resizeX = T[0]
        resizeY = T[1]

        self.worldNamespace['imgWidth'] = str(resizeX)
        self.worldNamespace['imgHeight'] = str(resizeY)
        self.worldNamespace['ground_plane_material_path'] = material_path



    def changeRobotAndWorld(self):
        """
        This changes the robot in the launch file
        """
        # Accomplish through edits in the launch file
        path = "/home/dongyanqi/catkin_ws/src/simulator_gazebo/launch/simulation_launch/" + self.spec_file_name + "_world.launch"

        # Open launch file template
        f = open(self.tmpl_path + 'world.launch.tmpl', 'r')
        launchDef = f.read()
        launchNamespace = {}
        f.close()

        launchNamespace['robot_pkg'] = self.package
        launchNamespace['robot_launch'] = self.launchFile

        # change robot position
        pos_str = os.getenv('ROBOT_INITIAL_POSE')
        if not pos_str is None:

            pos_x = pos_str.split()[1]
            pos_y = pos_str.split()[3]
            # pos_x = str(1)
            # pos_y = str(0)

            launchNamespace['coord_x'] = pos_x
            launchNamespace['coord_y'] = pos_y

        # World file
        launchNamespace['world_file'] = self.worldFile

        # Create launch File
        launch = Template(launchDef, searchList=launchNamespace)
        f = open(path, 'w+')
        f.write(str(launch))
        f.close()



    def addObstacles(self):
        # check if there are obstacles. If so, they will be added to the world
        # square obstacle limited
        for region in self.original_regions.regions:
            if region.isObstacle is True:
                addObstacle = True
                break

        if addObstacle is False:
            self.worldNamespace["OBSTACLE"] = False
            print("INIT:NO obstacle")

        if addObstacle is True:
            print("INIT:OBSTACLES!!")
            self.worldNamespace["OBSTACLE"] = True
            self.worldNamespace["obstacle_count"] = 0
            self.worldNamespace["pos_x"] = []
            self.worldNamespace["pos_y"] = []
            self.worldNamespace["lengths"] = []
            self.worldNamespace["depths"] = []
            self.worldNamespace["heights"] = []
            ######### ADDED

            self.map = {'polygon': {}, 'original_name': {}, 'height': {}}
            for region in self.proj.rfi.regions:
                self.map['polygon'][region.name] = self.createRegionPolygon(region)
                for n in range(len(region.holeList)):  # no of holes
                    self.map['polygon'][region.name] -= self.createRegionPolygon(region, n)

            ###########
            for region in self.original_regions.regions:
                if region.isObstacle is True:
                    poly_region = self.createRegionPolygon(region)
                    center = poly_region.center()
                    print("center:" + str(center), file=sys.stdout)
                    pose = self.coordmap_map2lab(region.getCenter())
                    print("pose:" + str(pose), file=sys.stdout)
                    pose = center
                    height = region.height
                    if height == 0:
                        height = self.original_regions.getMaximumHeight()

                    a = poly_region.boundingBox()
                    size = [a[1] - a[0], a[3] - a[2]]  # xmax,xmin,ymax,ymin

                    # if "pillar" in region.name.lower():  # cylinders
                    #     radius = min(size[0], size[1]) / 2
                    #     print("INIT: pose " + str(pose) + " height: " + str(height) + " radius: " + str(radius))
                    #     self.addCylinder(i, radius, height, pose)
                    # else:
                    #     length = size[0]  # width in region.py = size[0]
                    #     depth = size[1]  # height in region.py = size[1]
                    #     print("INIT: pose " + str(pose) + " height: " + str(height) + " length: " + str(
                    #         length) + " depth: " + str(depth))
                    #     self.addBox(i, length, depth, height, pose)

                    length = size[0]  # width in region.py = size[0]
                    depth = size[1]  # height in region.py = size[1]
                    # print("INIT: pose " + str(pose) + " height: " + str(height) + " length: " + str(length) + " depth: " + str(depth))

                    self.worldNamespace["pos_x"].append(pose[0])
                    self.worldNamespace["pos_y"].append(pose[1])
                    self.worldNamespace["lengths"].append(length)
                    self.worldNamespace["depths"].append(depth)
                    self.worldNamespace["heights"].append(height)
                    self.worldNamespace["obstacle_count"] += 1


    def createBoundary(self):
        boundary = None
        for region in self.original_regions.regions:
            if region.name == "boundary":
                boundary = region

        self.worldNamespace["boundary_count"] = 0
        self.worldNamespace["boundary_pos_x"] = []
        self.worldNamespace["boundary_pos_y"] = []
        self.worldNamespace["boundary_lengths"] = []
        self.worldNamespace["boundary_depths"] = []
        self.worldNamespace["boundary_heights"] = []


        try:
            poly_boundary = self.createRegionPolygon(boundary)
            bbox = poly_boundary.boundingBox()  # xmin, xmax, ymin and ymax
            points = [regions.Point(bbox[0], bbox[2]), regions.Point(bbox[1], bbox[2]), regions.Point(bbox[1], bbox[3]), regions.Point(bbox[0], bbox[3])]
            # points = map(self.coordmap_map2lab, points)
            points = [(pt.x, pt.y) for pt in points]
            points = points + points[:1]

            for i in range(1, len(points)):
                pos_x = (points[i][0] + points[i-1][0])/2
                pos_y = (points[i][1] + points[i-1][1])/2

                length = abs(points[i][0] - points[i-1][0]) if abs(points[i][0] - points[i-1][0]) >= 1e-3 else 0.05
                depth = abs(points[i][1] - points[i-1][1]) if abs(points[i][1] - points[i-1][1]) >= 1e-3 else 0.05
                height = 1

                self.worldNamespace["boundary_pos_x"].append(pos_x)
                self.worldNamespace["boundary_pos_y"].append(pos_y)
                self.worldNamespace["boundary_lengths"].append(length)
                self.worldNamespace["boundary_depths"].append(depth)
                self.worldNamespace["boundary_heights"].append(height)
                self.worldNamespace["boundary_count"] += 1

        except Exception as e:
            print(e)



    def rosSubProcess(self, proj, worldFile='ltlmop_map.world'):
        start = subprocess.Popen(['roslaunch simulator_gazebo ' + self.spec_file_name + "_world.launch"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        start_output = start.stdout
        # logging.info(start_output.read())

        time.sleep(5)
        logging.info("Launch finished")

        # # Wait for it to fully start up
        # while 1:
        #     logging.info("Test")
        #     input = start_output.readline()
        #     logging.info("Test")
        #     print(type(input))
        #     logging.info("Get input: " + input)
        #     if input == '': # EOF
        #         print("(INIT) WARNING:  Gazebo seems to have died!")
        #         break
        #     if "Successfully spawned" or "successfully spawned" in input:
        #         # Successfully spawend is output from the creation of the PR2
        #         # It might get stuck waiting for another type of robot to spawn
        #         logging.info("Successfully spawned")
        #         time.sleep(5)
        #         break

    def centerTheRobot(self, executor, init_region):
        # Start in the center of the defined initial region

        try: #Normal operation
            initial_region = self.proj.rfiold.regions[self.proj.rfiold.indexOfRegionWithName(init_region)]
        except: #Calibration
            initial_region = self.proj.rfi.regions[self.proj.rfi.indexOfRegionWithName(init_region)]
        center = initial_region.getCenter()

        # Load the map calibration data and the region file data to feed to the simulator
        coordmap_map2lab, coordmap_lab2map = executor.hsub.getMainRobot().getCoordMaps()
        map2lab = list(coordmap_map2lab(array(center)))

        print("Initial region name: " + str(initial_region.name) + " I think I am here: " + str(map2lab) + " and center is: " + str(center))

        os.environ['ROBOT_INITIAL_POSE'] = "-x "+str(map2lab[0])+" -y "+str(map2lab[1])
        # os.environ['ROBOT_INITIAL_POSE'] = "-x " + str(0) + " -y " + str(0)
        # print((map2lab[0], map2lab[1]))
