
import sys
import os
import math


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
SCS_MOVING_SPEED            = 900  #600      # SCServo moving speed
SCS_MOVING_ACC              = 30   #20       # SCServo moving acc
BAUDRATE                    = 1000000          # SCServo default baudrate : 1000000

# Default setting

SERVO_1_POS_LIMIT = [970, 3020]         # вращение 180deg
SERVO_2_POS_LIMIT = [1000, 1853]         # вперед - назад 75deg
SERVO_3_POS_LIMIT = [1365, 2048]         # вниз - вверх 60deg

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


def set_position(position):
    angles = inverse_kinematic((position))
    
    res_pos=list(forward_kinematics(angles))
    for i in range(3):
        res_pos[i]=int(res_pos[i])
    print("   ",res_pos)

    rotate_pulse, left_pulse, right_pulse=deg_to_pulse(angles)
    #print(rotate_pulse, left_pulse, right_pulse)

    packetHandler.WritePosEx(1, rotate_pulse, SCS_MOVING_SPEED, SCS_MOVING_ACC) # вращение 
    packetHandler.WritePosEx(2, left_pulse, SCS_MOVING_SPEED, SCS_MOVING_ACC) # вперед-назад
    packetHandler.WritePosEx(3, right_pulse, SCS_MOVING_SPEED, SCS_MOVING_ACC) # вверх-вниз

def set_position_pulses(pulses):
    rotate, left, right=pulses

    #rotate_angle,left_angle, right_angle = kinematics.pulse_to_deg((rotate,left,right))
    #res_pos=list(justkinematics.forward_kinematics((rotate_angle,left_angle,right_angle)))
    #for i in range(3):
    #    res_pos[i]=int(res_pos[i])
    #print("   ",res_pos)

    packetHandler.WritePosEx(1, rotate, SCS_MOVING_SPEED, SCS_MOVING_ACC) # вращение 
    packetHandler.WritePosEx(2, left, SCS_MOVING_SPEED, SCS_MOVING_ACC) # вперед-назад
    packetHandler.WritePosEx(3, right, SCS_MOVING_SPEED, SCS_MOVING_ACC) # вверх-вниз

def val_map(x, in_min, in_max, out_min, out_max):
    return round((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min,2)

def pulse_to_deg(pulses):
    rotate_pulse, left_pulse, right_pulse= pulses
    #диапазон вращения сервы 1 - основание в градусах от 30 до 210 градусов, 120 это центр и по оси X=0
    rotate_angle=val_map(rotate_pulse, SERVO_1_POS_LIMIT[0] , SERVO_1_POS_LIMIT[1], 30, 210)
    #диапазон вращения сервы 2, отвечающей за вперед-назад, в градусах от 90 до 135 градусов
    left_angle=val_map(left_pulse, SERVO_2_POS_LIMIT[0] , SERVO_2_POS_LIMIT[1], 90, 165)
    #диапазон вращения сервы 3, отвечающей за вверх-вниз, в градусах от 60 до 0 градусов
    right_angle=val_map(right_pulse, SERVO_3_POS_LIMIT[0] , SERVO_3_POS_LIMIT[1],60, 0)
    return rotate_angle, left_angle, right_angle

def deg_to_pulse(angles):
    rotate_angle, left_angle, right_angle= angles
    #диапазон вращения сервы 1 - основание в градусах от 30 до 210 градусов, 120 это центр и по оси X=0
    rotate_pulse=int(val_map(rotate_angle, 30, 210, SERVO_1_POS_LIMIT[0] , SERVO_1_POS_LIMIT[1]))
    #диапазон вращения сервы 2, отвечающей за вперед-назад, в градусах от 90 до 135 градусов
    left_pulse=int(val_map(left_angle, 90, 165, SERVO_2_POS_LIMIT[0] , SERVO_2_POS_LIMIT[1]))
    #диапазон вращения сервы 3, отвечающей за вверх-вниз, в градусах от 50 до 0 градусов
    right_pulse=int(val_map(right_angle, 60, 0, SERVO_3_POS_LIMIT[0] , SERVO_3_POS_LIMIT[1]))
    return rotate_pulse, left_pulse, right_pulse

L0 = 70 #84.4 # высота от стола до плоскости вращения основания
L1 = 8.14 # 
L2 = 128.4 # длина плеча левого мотора перпендикулярно вверх
L3 = 138.0 # длина от вершины прямого угла до оси end-of-tool
L4 = 40 #16.8 # высота по Z инструмента ???

RAD_PER_DEG = math.pi / 180
DEG_PER_RAD = 180 / math.pi
DOUBLE_PI = math.pi * 2


def forward_kinematics(angles):
    """
    JetMax forward kinematics
    @param angles: active angles [rotate, angle left, angle right]
    @return: end point position (x, y, z)
    """

    alpha1, alpha2, alpha3 = [angle * RAD_PER_DEG for angle in angles]
    alpha1 += 150 * RAD_PER_DEG
    alpha1 = alpha1 - DOUBLE_PI if alpha1 > DOUBLE_PI else alpha1
    beta = alpha2 - alpha3
    side_beta = math.sqrt(L2 ** 2 + L3 ** 2 - 2 * L2 * L3 * math.cos(beta))
    cos_gamma = ((side_beta ** 2 + L2 ** 2) - L3 ** 2) / (2 * side_beta * L2)
    cos_gamma = cos_gamma if cos_gamma < 1 else 1
    gamma = math.acos(cos_gamma)
    alpha_gamma = math.pi - alpha2
    alpha = alpha_gamma - gamma
    z = side_beta * math.sin(alpha)
    r = math.sqrt(side_beta ** 2 - z ** 2)
    z = z + L0-L4
    r = r + L1
    x = r * math.cos(alpha1)
    y = r * math.sin(alpha1)*(-1) #added -1 to rotate axis
    return round(x,1), round(y,1), round(z,1)


def inverse_kinematic(position):
    """
    JetMax inverse kinematics
    @param position: target position (x, y, z)
    @return: joint angles list
    """
    x, y, z = position
    y=y*(-1) #added -1 to rotate axis
    z=z+L4
    r = math.sqrt(x ** 2 + y ** 2)
    if x == 0:
        theta1 = math.pi / 2 if y >= 0 else math.pi / 2 * 3  # pi/2 90deg, (pi * 3) / 2  270deg
    else:
        if y == 0:
            theta1 = 0 if x > 0 else math.pi
        else:
            theta1 = math.atan(y / x)  # θ=arctan(y/x) (x!=0)
            if x < 0:
                theta1 += math.pi
            else:
                if y < 0:
                    theta1 += math.pi * 2

    r = r - L1
    z = z - L0
    if math.sqrt(r ** 2 + z ** 2) > (L2 + L3):
        raise ValueError('Unreachable position: x:{}, y:{}, z:{}'.format(x, y, z))

    alpha = math.atan(z / r)
    beta = math.acos((L2 ** 2 + L3 ** 2 - (r ** 2 + z ** 2)) / (2 * L2 * L3))
    gamma = math.acos((L2 ** 2 + (r ** 2 + z ** 2) - L3 ** 2) / (2 * L2 * math.sqrt(r ** 2 + z ** 2)))

    theta1 = theta1
    theta2 = math.pi - (alpha + gamma)
    theta3 = math.pi - (beta + alpha + gamma)

    theta1 = theta1 * DEG_PER_RAD
    if 30 < theta1 < 150:  # The servo motion range is 240 deg. 150~360+0~30 = 240
        raise ValueError('Unreachable position: x:{}, y:{}, z:{}'.format(x, y, z))
    theta1 = theta1 + 360 if theta1 <= 30 else theta1  # 0~360 to 30~390
    theta1 = theta1 - 150
    return theta1, theta2 * DEG_PER_RAD, theta3 * DEG_PER_RAD