# vehicle-tracker - By: Manuel - jue. feb. 20 2020

import sys
from classes import *
import sensor
import image
import lcd
import time
import KPU as kpu
import math
import utime
from machine import UART
from Maix import FPIOA
#from Maix import GPIO
clock = time.clock()

def sendReport(event):
    if(event == 'in'):
        uart.write('AT$FORM=0,@T,0,"%VB"\r\n')
        uart.write('AT$FUNC="VRBL",0,151\r\n')
        uart.write('AT$GPOS=2,0\r\n')
    elif(event == 'out'):
        uart.write('AT$FORM=0,@T,0,"%VB"\r\n')
        uart.write('AT$FUNC="VRBL",0,150\r\n')
        uart.write('AT$GPOS=2,0\r\n')

#---- main() ----#

lcd.init(freq=15000000)
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_vflip(0)
sensor.run(1)

fpioa = FPIOA()
fpioa.help(fpioa.UART2_TX)
fpioa.help(fpioa.UART2_RX)
fpioa.set_function(12, fpioa.UART2_TX)
fpioa.set_function(11, fpioa.UART2_RX)
fpioa.set_function(10, fpioa.GPIO0)

uart = UART(UART.UART2,57600,8,None,1,timeout=1000,read_buf_len=4096)
write_str = 'Received\n'

"""
while(True):
     a = uart.readline()
     if(a != None):
        print(a)
        uart.write(write_str)
"""

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
task = kpu.load(0x500000)
anchor = (1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52)
a = kpu.init_yolo2(task, 0.5, 0.3, 5, anchor)

vehiclePath = path(5,100.0)
vehicleWatcher = watcher({'x':[0, 60], 'y': [0, 200]},{'x':[190, 240], 'y': [0, 200]})

while(True):
    clock.tick()
    img = sensor.snapshot()
    code = kpu.run_yolo2(task, img)
    #print(clock.fps())
    vehiclePath.watchDog()
    if code:
        for i in code:
            box = img.draw_rectangle(i.rect())
            lcd.draw_string(
                i.x(), i.y(), str(i.index()), lcd.RED, lcd.WHITE)
            lcd.draw_string(i.x(), i.y()+12, '%f1.3' %
                            i.value(), lcd.RED, lcd.WHITE)
            if(i.classid() == 5 or i.classid() == 6 or i.classid() == 13):
                vehiclePath.insertPoint([i.x(), i.y()])
                vehiclePath.feedWatchDog()
                event = vehicleWatcher.watch(vehiclePath)
                if event != 'none':
                    print(event)
                    sendReport(event)
    #print(vehiclePath.getPath(), vehicleWatcher.gotOut(vehiclePath), vehicleWatcher.gotIn(vehiclePath))
    lcd.display(img)

kpu.deinit(task)





















