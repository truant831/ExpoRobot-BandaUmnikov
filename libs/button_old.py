import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
but_pin = 15
GPIO.setup(but_pin, GPIO.IN)

def btn_pressed():
    is_pressed=not GPIO.input(but_pin)
    return is_pressed

def clean_btn():
    GPIO.cleanup()

def wait_time_or_btn(timeout=3):
    start_time=time.time()
    is_pressed=not GPIO.input(but_pin)

    while not is_pressed:
        is_pressed=not GPIO.input(but_pin)
        elapsed_time=time.time()-start_time

        if elapsed_time>timeout:
            print("Button wait time is over. was not pressed")
            break
        time.sleep(0.1)

    return is_pressed
