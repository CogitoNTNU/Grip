import threading
import time

import serial

ser = serial.Serial("/dev/ttyAMA0", 9600, timeout=0.5)


def reader():
    while True:
        line = ser.readline()
        if line:
            print("Arduino says:", line.decode(errors="ignore").strip())


def writer():
    while True:
        ser.write(b"Hello Arduino!\n")
        time.sleep(2)


# start threads
threading.Thread(target=reader, daemon=True).start()
threading.Thread(target=writer, daemon=True).start()

while True:
    time.sleep(1)
