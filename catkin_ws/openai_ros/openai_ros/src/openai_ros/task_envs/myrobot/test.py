import rospy
from gazebo_msgs.srv import *
from transforms3d.euler import quat2euler

if __name__ == "__main__":
    rospy.wait_for_service('/gazebo/get_model_state')
    try:
        gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        request = GetModelStateRequest()
        # print(self.model_name)
        request.model_name = "turtlebot3_burger"
        model_state = gms(request)
        #Cartesian Pose
        # self.robot_pos_x = model_state.pose.position.x
        # self.robot_pos_y = model_state.pose.position.y
        # self.robot_pos_z = model_state.pose.position.z
        #Quaternion Orientation
        or_x = model_state.pose.orientation.x
        or_y = model_state.pose.orientation.y
        or_z = model_state.pose.orientation.z
        or_w = model_state.pose.orientation.w
        angles = quat2euler([or_w, or_x, or_y, or_z])
        print(angles)
        # self.robot_or_alpha = angles[0]
        # self.robot_or_beta = angles[1]
        # self.robot_or_theta = angles[2]
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
