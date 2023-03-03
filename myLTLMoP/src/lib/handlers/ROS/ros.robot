RobotName: # The name of the robot
ROS

Type: # Robot type
ROS

InitHandler: # Robot default init handler with default argument values
ROS.RosInitHandler(worldFile="ltlmop_map.world", robotPixelWidth=200, robotPhysicalWidth=.5, robotPackage="pr2_gazebo", robotLaunchFile="pr2.launch", modelName = "pr2")

PoseHandler: # Robot default pose handler with default argument values
ROS.RosPoseHandler(modelName="pr2")

SensorHandler: # Robot default sensors handler with default argument values
ROS.RosSensorHandler()

ActuatorHandler: # Robot default actuator handler wit hdefault argument values
ROS.RosActuatorHandler()

MotionControlHandler: # Robot default motion control handler with default argument values
share.MotionControl.HeatControllerHandler()

DriveHandler: # Robot default drive handler with deafult argument values
ROS.RosDriveHandler(d=.6)

LocomotionCommandHandler: # Robot default locomotion command handler with default argument values
ROS.RosLocomotionCommandHandler(velocityTopic='/base_controller/command')
