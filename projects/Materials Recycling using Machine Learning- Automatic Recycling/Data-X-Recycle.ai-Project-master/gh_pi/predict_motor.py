'''
    Functions to move motors and light LEDs 
'''

import RPi.GPIO as GPIO
from Adafruit_MotorHAT import Adafruit_MotorHAT
import atexit
from time import sleep

# turn off all motors
def turnOffMotors():
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)
    print("Motors released")

# motor parameters
mh = Adafruit_MotorHAT(addr=0x60)
atexit.register(turnOffMotors)
myStepper = mh.getStepper(200, 1)
myStepper.setSpeed(200)

def led_blue():

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(11, GPIO.OUT)  # RGB red
    GPIO.output(11, GPIO.LOW)
    GPIO.setup(13, GPIO.OUT)  # RGB green
    GPIO.output(13, GPIO.LOW)
    GPIO.setup(15, GPIO.OUT)  # RGB blue
    GPIO.output(15, GPIO.LOW)
    chan_list = (11, 13, 15)
    GPIO.output(chan_list, (GPIO.LOW, GPIO.LOW, GPIO.HIGH))
    sleep(6)
    GPIO.cleanup()

def led_red():

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(11, GPIO.OUT)  # RGB red
    GPIO.output(11, GPIO.LOW)
    GPIO.setup(13, GPIO.OUT)  # RGB green
    GPIO.output(13, GPIO.LOW)
    GPIO.setup(15, GPIO.OUT)  # RGB blue
    GPIO.output(15, GPIO.LOW)
    chan_list = (11, 13, 15)
    GPIO.output(chan_list, (GPIO.HIGH, GPIO.LOW, GPIO.LOW))
    sleep(6)
    GPIO.cleanup()

def led_yellow():

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(11, GPIO.OUT)  # RGB red
    GPIO.output(11, GPIO.LOW)
    GPIO.setup(13, GPIO.OUT)  # RGB green
    GPIO.output(13, GPIO.LOW)
    GPIO.setup(15, GPIO.OUT)  # RGB blue
    GPIO.output(15, GPIO.LOW)
    chan_list = (11, 13, 15)
    GPIO.output(chan_list, (GPIO.HIGH, GPIO.HIGH, GPIO.LOW))  
    sleep(6)
    GPIO.cleanup()

def led_green():

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(11, GPIO.OUT)  # RGB red
    GPIO.output(11, GPIO.LOW)
    GPIO.setup(13, GPIO.OUT)  # RGB green
    GPIO.output(13, GPIO.LOW)
    GPIO.setup(15, GPIO.OUT)  # RGB blue
    GPIO.output(15, GPIO.LOW)
    chan_list = (11, 13, 15)
    GPIO.output(chan_list, (GPIO.LOW, GPIO.HIGH, GPIO.LOW))
    sleep(6)
    GPIO.cleanup()

def move_recycle():

    myStepper.step(200, Adafruit_MotorHAT.FORWARD, Adafruit_MotorHAT.DOUBLE)
    sleep(1)
    myStepper.step(200, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.DOUBLE)

def move_compost():

    myStepper.step(200, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.DOUBLE)
    sleep(1)
    myStepper.step(200, Adafruit_MotorHAT.FORWARD, Adafruit_MotorHAT.DOUBLE)

def move_landfill():

    myStepper.step(40, Adafruit_MotorHAT.FORWARD, Adafruit_MotorHAT.DOUBLE)
    sleep(1)
    myStepper.step(40, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.DOUBLE)

  

