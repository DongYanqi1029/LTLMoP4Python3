RobotName: # The name of the robot
ROS

Type: # Robot type
ROS

InitHandler: # Robot default init handler with default argument values
ROS.RosInitHandler(worldFile='ltlmop_map.world', robotPixelWidth=40, robotPhysicalWidth=.5, robotPackage="simulator_gazebo", robotLaunchFile="turtlebot3_burger.launch", modelName = "turtlebot3_burger")

PoseHandler: # Robot default pose handler with default argument values
ROS.RosPoseHandler(modelName="turtlebot3_burger")

SensorHandler: # Robot default sensors handler with default argument values
ROS.RosSensorHandler()

ActuatorHandler: # Robot default actuator handler wit default argument values
ROS.RosActuatorHandler()

MotionControlHandler: # Robot default motion control handler with default argument values
# share.MotionControl.HeatControllerHandler()
# share.MotionControl.VectorControllerHandler()
# ROS.ROSMotionControlHandler(robot_type=3, max_angle_goal=6.28, max_angle_overlap=1.57, plotting=True)
# share.MotionControl.RRTControllerHandler(robot_type=3, max_angle_goal=6.28, max_angle_overlap=1.57, plotting=True)
# ROS.RRTandTD3ControllerHandler(model_name='TD3_MyRobotWorld-v0_actor', model_path='/home/dongyanqi/catkin_ws/src/TD3_UGV_openai_ros/models', robot_type=3, max_angle_goal=6.28, max_angle_overlap=1.57, plotting=False)
ROS.DRLControllerHandler(model_name='TD3_MyRobotWorld-v0_actor', model_path='/home/dongyanqi/catkin_ws/src/TD3_UGV_openai_ros/models')


DriveHandler: # Robot default drive handler with default argument values
ROS.RosDriveHandler(d=.6)

LocomotionCommandHandler: # Robot default locomotion command handler with default argument values
ROS.RosLocomotionCommandHandler(velocityTopic='/cmd_vel')
