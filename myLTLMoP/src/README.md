# **myLTLMoP**

## **任务场景**
- Single agent 

## **常见修改**
- dict.keys()在python3中并不是列表
- iteritems() -> items()
- print
- UDPSocket.recvFrom sendTo stdin.write 对应的encode与decode
- xrange -> range

## Handlers

- InitHandler 负责地图信息的初始化，包括Gazebo世界的构建以及对应Simulation的启动
- PoseHandler 负责仿真中机器人的位置及姿态信息的获取，通过Ros Topic 获取
- DriveHandler 通过 setVelocity 函数设置机器人的移动速度
- LocomotionCommandHandler 根据 DriveHandler 提供的速度信息，将移动指令传递给对应的 Ros Topic
- MotionControlHandler 通常内置机器人的运动控制算法，通过 Drivehandler 的函数API来实现车辆的控制。其中本Handler的gotoRegion方法是重点。
- Sensor/ActuatorHandler 管理机器人的sensor/actuator命题，类中的每一个方法对应一个命题的处理。

## **Sensors & Actuators**
- 使用hsub中的getSensorValue方法得到sensor命题值(调用方法，后得到命题返回值)
- executeStrategy updateOutputs->hsub.setActuactorValue(更新命题值，后调用方法)
- 不支持instantaneous action

## **Proposition Mapping**
- 在命题映射中，软件使用handler对应方法的备注来得到方法的参数类型等信息
- sensor与actuator的命题映射函数在对应的hander方法中寻找

## **待实现功能**
- 强化学习能力改进
- 优化网络模型的保存与读取方式
