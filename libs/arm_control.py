
import sys
import os

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
        
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

sys.path.append("..")
from scservo_sdk import *                 # Uses SCServo SDK library

DEVICENAME                  = '/dev/ttyUSB0'    # Check which port is being used on your controller                                            
SCS_MOVING_SPEED            = 600        # SCServo moving speed
SCS_MOVING_ACC              = 20          # SCServo moving acc
BAUDRATE                    = 1000000          # SCServo default baudrate : 1000000

# Default setting

SERVO_1_POS_LIMIT = [970, 3020]         # вращение
SERVO_2_POS_LIMIT = [1000, 1500]         # вперед - назад
SERVO_3_POS_LIMIT = [1425, 2073]         # вниз - вверх


portHandler = PortHandler(DEVICENAME)
packetHandler = sms_sts(portHandler)
    
# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()


def set_pulse_servo(id,pulse):
    packetHandler.WritePosEx(id, int(pulse), SCS_MOVING_SPEED, SCS_MOVING_ACC) 

def set_position(pulses):
    servo1,servo2,servo3=pulses
    set_pulse_servo(1,servo1)
    set_pulse_servo(2,servo2)
    set_pulse_servo(3,servo3)