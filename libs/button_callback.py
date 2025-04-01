
import __main__
from libs.button import *
from libs.play_sounds import Say_phraze
import RPi.GPIO as GPIO
import time



P_BUTTON = 15 # adapt to your wiring
is_pressed=False

def setup():
    GPIO.setmode(GPIO.BOARD)
    button = Button(P_BUTTON) 
    button.addXButtonListener(onButtonEvent)

def onButtonEvent(button, event):
    global is_pressed
    if event == BUTTON_PRESSED:
        print ("pressed")
        is_pressed=True
    elif event == BUTTON_RELEASED:
        print ("released") 
        is_pressed=False    
    elif event == BUTTON_LONGPRESSED:
       print ("long pressed")
    elif event == BUTTON_CLICKED:
        print ("clicked")
        if not __main__.match_found: #не засчитываем щелчок если сейчас роботт в состоянии "нашел пару
            __main__.was_clicked=True
    elif event == BUTTON_DOUBLECLICKED:
        __main__.was_dbl_clicked=True
        print ("double clicked")

def clean_btn():
    GPIO.cleanup()

def wait_time_or_btn(timeout=3):
    global is_pressed
    start_time=time.time()
    #is_pressed=not GPIO.input(P_BUTTON )

    while not is_pressed:
        #is_pressed=not GPIO.input(P_BUTTON)
        elapsed_time=time.time()-start_time

        if elapsed_time>timeout:
            print("Button wait time is over. was not pressed")
            break
        time.sleep(0.1)
    return is_pressed

setup()