# README

## **1. myLTLMoP**

### **1.1. 任务场景**

- Single agent 

### **1.2. 常见修改**

- dict.keys()在python3中并不是列表
- iteritems() -> items()
- print
- UDPSocket.recvFrom sendTo stdin.write 对应的encode与decode
- xrange -> range

### 1.3. Handlers

- InitHandler 负责地图信息的初始化，包括Gazebo世界的构建以及对应Simulation的启动
- PoseHandler 负责仿真中机器人的位置及姿态信息的获取，通过Ros Topic 获取
- DriveHandler 通过 setVelocity 函数设置机器人的移动速度
- LocomotionCommandHandler 根据 DriveHandler 提供的速度信息，将移动指令传递给对应的 Ros Topic
- MotionControlHandler 通常内置机器人的运动控制算法，通过 Drivehandler 的函数API来实现车辆的控制。其中本Handler的gotoRegion方法是重点。
- Sensor/ActuatorHandler 管理机器人的sensor/actuator命题，类中的每一个方法对应一个命题的处理。

### **1.4. Sensors & Actuators**

- 使用hsub中的getSensorValue方法得到sensor命题值(调用方法，后得到命题返回值)
- executeStrategy updateOutputs->hsub.setActuactorValue(更新命题值，后调用方法)
- 不支持instantaneous action

### **1.5. Proposition Mapping**

- 在命题映射中，软件使用handler对应方法的备注来得到方法的参数类型等信息
- sensor与actuator的命题映射函数在对应的hander方法中寻找

### **1.6. 待实现功能**

- 强化学习能力改进
- 优化网络模型的保存与读取方式

## 2. catkin_ws

### 2.1. TD3_UGV_openai_ros

本文件夹为一个catkin package

#### 2.1.1. setup.py

本文件可以将对应的 ros catkin package 在 catkin_make 的时候编译成可调用的 python package (在此之前需要运行catkin_workspace中的setup.bash)

#### 2.1.2. config/

该文件夹中存放着一些配置文件，它们中存储的信息一般用于定义python脚本中的超参数信息等

#### 2.1.3. launch/

该文件夹中存放着 ros launch文件，用于启动模型训练、测试等python脚本

#### 2.1.4. scripts/

存放python脚本

### 2.2. openai_ros

src: https://bitbucket.org/theconstructcore/openai_ros.git

catkin package

#### 2.2.1 robot_envs/

描述Agent的信息，比如机器人的移动、感知信息获取、动作执行等API函数的描述。

#### 2.2.2 task_env/

描述任务环境的信息，比如任务的目标、奖励函数的设置等。新建一个 task env 后需要在task_env_lists.py文件中进行添加。

### 2.3. simulator_gazebo

catkin package, 包含描述gazebo仿真信息的文件

#### 2.3.1 launch/DRL_launch

与openai_ros任务环境(task_env)对应的launch启动文件，用于启动gazebo仿真环境

#### 2.3.2 launch/robot_launch

与openai_ros机器人环境(robot_env)对应的launch启动文件，用于在gazebo环境中唤醒机器人

#### 2.3.2 launch/simulation_launch

- LTLMoP任务环境的Gazebo仿真模拟启动文件
- 使用键盘控制机器人的launch文件
- rviz启动launch文件

## 3. configs/

存储环境配置文件，如conda虚拟环境

```sh
conda env export > [yaml_name].yaml
conda env create -f [yaml_name].yaml
```

