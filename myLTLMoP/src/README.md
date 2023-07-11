# **myLTLMoP**

## **任务场景**
- Single-agent patrolling

## **常见修改**
- dict.keys()在python3中并不是列表
- iteritems() -> items()
- print
- UDPSocket.recvFrom sendTo stdin.write 对应的encode与decode
- xrange -> range

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
