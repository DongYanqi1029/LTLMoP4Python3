# **myLTLMoP**

## **Multi-agent Simulation**

- 开启多个specEditor来实现多个机器人
- 确保多个机器人都是使用的同一个region map
- 根据region map来实现world文件的编写
- 修改initHnadler的编写，做到模块化，使得world文件由region map决定，机器人的加入在launch文件中体现
- 使用cheetah进行world以及launch文件的模板化自动生成