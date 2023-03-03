# named pipe Client
#encoding: utf-8
import sys
import os
import time
import re
#import automaton
from distutils.util import strtobool
import copy


write_path = "/home/diy/UAS_in.pipe"
read_path = "/home/diy/UAS_out.pipe"

r1 = [('bit0', False), ('bit1', True)]
r2 = [('bit0', False), ('bit1', False)]
r3 = [('bit0', True), ('bit1', False)]

outFlag = 0 #防止无限输出mode_circle导致无法启动
go_r1 = "guided -35.3633143 149.1651910 3"  
go_r2 = "guided -35.3632490 149.1651758 3" 
go_r3 = "guided -35.3632079 149.1651925 3"   


sensors = []
actuators = []
regions = []
lastInput = []
position = []
sensorState = []
lastPosition = []
State=[]
curNodeNum = 0
pos = []
curr_state = None
lineNo = 0  #读写标记
#filepath = os.path.abspath("/home/diy/Desktop/SecurityProperty/LTLMoP+Example/GPSattack/monitor/monitor.aut")
#filepath = os.path.abspath("/home/diy/Desktop/SecurityProperty/LTLMoP+Example/test/UAV1.aut")
#filepath = os.path.abspath("/home/diy/Desktop/SecurityProperty/LTLMoP+Example/test2/main.aut")
filepath = os.path.abspath("/home/diy/Desktop/SecurityProperty/LTLMoP+Example/test4/main.aut")


def match(filepath):
    f = open(filepath,'r')
    matchline = ''
    for i in range(10):
        matchline = f.readline()
        if matchline.startswith('State'):
            break
    props = matchline.split("->")[1].strip().strip("<").strip(">").strip("\n").replace(":0","").replace(":1","").split(", ")
    print 'props:',props
    f.close()


    a = open(r'/home/diy/Desktop/vmFiles/LTLMoP_ROS_Ardupilot/src/mycode/sensorlist.txt', 'r')
    sensorline = a.read()
    for p in props:
        if p in sensorline.strip("\n").split(" "):
            sensors.append(p)
    print 'sensors:',sensors
    a.close()

    b = open(r'/home/diy/Desktop/vmFiles/LTLMoP_ROS_Ardupilot/src/mycode/actuatorlist.txt', 'r')
    actuatorline = b.read()
    #print "props:",props
    #print actuatorline.strip("\n").split(" ")
    for pp in props:
        if pp in actuatorline.strip("\n").split(" "):
            actuators.append(pp)
    print 'actuators:',actuators
    b.close()

    c = open(r'/home/diy/Desktop/vmFiles/LTLMoP_ROS_Ardupilot/src/mycode/regionlist.txt', 'r')
    regionline = c.read()
    #print 'regionline:',regionline.strip("\n").split(" ")
    for r in props:
        r = r.replace(" ","")
        if r in regionline.strip("\n").split(" "):
            regions.append(r)
    print 'regions:',regions
    c.close()

    d = open(r'/home/diy/Desktop/vmFiles/LTLMoP_ROS_Ardupilot/src/mycode/position.txt', 'r')
    positionline = d.read()
    print positionline.split(" ")
    for p in positionline.split(" "):
        position.append(p.strip())
    print 'position:',position
    d.close()

def initState():
    global filepath
    global State
    f1 = open(filepath, 'r+')
    f1.seek(0)
    str = f1.read()
    f1.close()
    sens=str.strip().split("State ")[1:]
    
    for i in range(0,len(sens)):
        #print i,sens
        temp1= []
        temp2 = []
        temp3 = []
        temp4 = []
        temp5 = []
        node = []
        temp1.append(i)
        node.append(temp1)
        for s in sensors:
            x = sens[i].find(s)
            #print x
            #print "sensor:" ,bool(int(sens[i][x:x+len(s)+2].split(":")[1]))
            temp2.append(tuple((sens[i][x:x+len(s)+2].split(":")[0],bool(int(sens[i][x:x+len(s)+2].split(":")[1])))))
            #temp.append()
        #print "temp2" ,temp2
        node.append(temp2)
        for a in actuators:
            x = sens[i].find(a)
            #print x
            temp3.append(tuple((sens[i][x:x+len(a)+2].split(":")[0],bool(int(sens[i][x:x+len(a)+2].split(":")[1])))))
            #temp.append()
        #print "temp3" ,temp3
        node.append(temp3)

        for r in regions:
            x = sens[i].find(r)
            #print x
            temp4.append(tuple((sens[i][x:x+len(r)+2].split(":")[0],bool(int(sens[i][x:x+len(r)+2].split(":")[1])))))
            #temp.append()
            #print "temp4" ,temp4
        node.append(temp4)

        x = sens[i].find("With successors : ")  + len("With successors : ")
        for n in sens[i][x:].replace(" ",'').split(","):
            temp5.append(int(n))
        #print temp5
        node.append(temp5)

        #print 'node' , node
        State.append(node)
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print  State
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

def findNextNode():
    global State
    global curNodeNum
    global sensorState
    if curNodeNum < -1 or curNodeNum > len(State)-1:
        print "error node number"
        return -1
    else:
        for i in State[curNodeNum][4]:
            #print "sensorState in findNode",id(sensorState),sensorState
            if sensorState == State[i][1]:
                print "next node is :" ,i 
                return i
        print "error: can't find next node"
        return -1

def RefreshParm():
    f = open(r'/home/diy/Desktop/vmFiles/LTLMoP_ROS_Ardupilot/src/mycode/input.txt', 'r+')
    # f = open('input.txt', 'r+')
    global lineNo
    global sensorState
    global lastPosition
    global r1
    global r2
    global r3
    global outFlag

    f.seek(lineNo)
    line = None
    while line != "":
        #inputcmd = []
        line = f.readline()
        if line != "\n" and line != "":
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            line = line.strip(' ')
            print "line:",line
            if "r1, True" in line:
                outFlag = 1
            elif "r2, True" in line:
                outFlag = 1
            elif "r3, True" in line:
                outFlag = 1
                


            print "outFlag",outFlag
            cmd = line.split("\n")[0].split(",")[0].strip(' ').replace(" ",'')
            print "cmd:",cmd
            truth = line.split("\n")[0].split(",")[1].strip(' ')
            print "truth",truth
            print "sensorState before",id(sensorState),sensorState
            if cmd in sensors:
                if (cmd, bool(strtobool(truth))) in sensorState:
                    pass
                else:
                    sensorState[sensorState.index((cmd, bool(1-strtobool(truth))))] = (cmd, bool(strtobool(truth))) 
            print "sensorState after",id(sensorState),sensorState            
            if cmd in position:
                if cmd == "r1":
                    lastPosition = r1
                elif cmd == "r2":
                    lastPosition = r2
                elif cmd == "r3":
                    lastPosition = r3   
    lineNo = f.tell()
    f.close()


# def GetCmd(sensors, inputcmd):
#     f = open(r'/home/diy/Desktop/LTLMoP-test/src/mycode/input.txt', 'r+')
#     # f = open('input.txt', 'r+')
#     line = None
#     while line != "":
#         line = f.readline()
#         if line != "\n" and line != "":
#             line = line.strip()
#             cmd = line.split("\n")[0].split(",")[0].strip(' ')
#             truth = line.split("\n")[0].split(",")[1].strip(' ')
#             inputcmd.append((cmd, bool(truth)))
#     # f.seek(0)        
#     f.truncate()
#     f.close()
#     for sensor in sensors:
#         if not (sensor, True) in inputcmd:
#             inputcmd.append((sensor, False))

# def output(action):
#     for act in action:
#         act = ' '.join(str(act).split('_'))
#         # action = ''.join(action[2:-2])
#         print 'action:%s\n'%act
#         os.write( f, act )


def output(action):
    global outFlag
    for act in action:
        #print act
        (ata ,Truth) = act
        if Truth and outFlag:
            ata = ' '.join(ata.split('_'))
            print "output:",ata
            os.write( f, ata )


if __name__ == "__main__":

    # match(myautomaton_path)
    # # print sensors, regions, actuators
    # aut = automaton.Automaton(sensors, regions, actuators, myautomaton_path)
    # aut.build()
    # aut.print_graphviz('/home/diy/Desktop/LTLMoP-test/MDPItest/figure/automaton')
    # print 'builded'
    # init_state = aut.get_node(ID = 0)
    # curr_state = init_state

    match(filepath)
    initState()
    curNodeNum = 0
    sensorState = copy.deepcopy(State[0][1])
    #print "sensorState",id(sensorState),sensorState

    lastPosition = []

    lineNo = 0
    pos = copy.deepcopy(State[0][3])

    f = os.open( write_path, os.O_SYNC | os.O_CREAT | os.O_RDWR )
    time.sleep(0.5)



    while True:

        RefreshParm()
        if lastPosition:#检测是否到达初始点
            print "sensorState----------->>>," ,sensorState 
            if ('GUIDED', False) in sensorState: #非飞行状态进行状态切换 问题：非导航状态应该不会有pos变换
                print "('GUIDED', False) in input\n curr_state:",curNodeNum
                curNodeNum = findNextNode()
                pos = State[curNodeNum][3]
                #sensorState = curr_state.get_sensors()
                print "do action1"
                print "new curr_state:",curNodeNum
            else:#导航状态下进行位置切换
                if ('DOSattack',False) in sensorState:
                    if lastPosition == pos: #不换位置直接切换状态
                        print "\nlastPosition == pos\n"
                        print "curr_state:",curNodeNum
                        #sensorState = [('GUIDED', True), ('DOSattack', True)]
                        print "inputcp------------->",id(sensorState),sensorState
                        curNodeNum = findNextNode()
                        pos = State[curNodeNum][3]
                        print "do action2"
                        print "new curr_state:",curNodeNum
                    else:#换位置
                        print "go curr_state:",curNodeNum
                        if pos == r1:
                            print "go r1"
                            os.write( f, go_r1 )
                        elif pos == r2:
                            print "go r2"
                            os.write( f, go_r2 )
                        elif pos == r3:
                            print "go r3"
                            os.write( f, go_r3 )
                else:
                        print "curr_state:",curNodeNum
                        print "input",sensorState
                        curNodeNum = findNextNode()
                        pos = State[curNodeNum][3]
                        print "do action3"
                        print "new curr_state:",curNodeNum
                        if lastPosition != pos:
                            if pos == r1:
                                print "go r1"
                                os.write( f, go_r1 )
                            elif pos == r2:
                                print "go r2"
                                os.write( f, go_r2 )
                            elif pos == r3:
                                print "go r3"
                                os.write( f, go_r3 )
        else:
            print "not in initial position"
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        output(State[curNodeNum][2])
        time.sleep(5)
        # if not inputcmd:
        #     # os.write(f,'testing')
        #     print 'no inputcmd'
        #     time.sleep(5)
        #     continue
        # # print riskycmd
        # (action, curr_state) = interpreter.interpreter(aut, curr_state, inputcmd)
        # if action is None:
        #     print 'no action'
        #     time.sleep(5)
        #     continue
        # # print "action:%s\n"%action



