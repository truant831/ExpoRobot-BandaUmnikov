import serial
import time

class RPiPico_serial():
    
    def __init__(self,p):
        self.port = serial.Serial(p, 115200) 
        self.nextData = 0
        self.dataInt = 0.001

    def apply(self, pomp, rgb):
        # if pomp_mode == "take":
        #     clap = 0 
        #     pomp = 1
        # elif pomp_mode == "hold":
        #     clap = 1
        #     pomp = 0
        # else:
        #     clap = 0
        #     pomp = 0
        if pomp == 1:
            clap = 0 
        else:
            clap = 0 #1
        msg=("dp"+str(pomp)+","+str(clap)+","+str(rgb))
        #print(msg)
        data = msg.encode('UTF-8')
        self.port.write(data)
        
    def stop(self):
        self.sendData("quit")