import serial, threading, time
from gpiozero import Servo

# Bruk BCM-numre (de du valgte)
servos = [
    Servo(17),
    Servo(27),
    Servo(22),
    Servo(23),
    Servo(24),
    Servo(25)
]

ser = serial.Serial("/dev/ttyAMA0", 9600, timeout=0.5)

def reader():
    while True:
        line = ser.readline()
        if line:
            try:
                value = int(line.decode(errors="ignore").strip())
                # Normaliser 0–1023 til -1..+1
                pos = (value / 1023) * 2 - 1
                pos = max(-1, min(1, pos))

                for s in servos:
                    s.value = pos

                print(f"[UART] Sensor={value:4d}  ->  Servo pos={pos:.2f}")
            except ValueError:
                pass

threading.Thread(target=reader, daemon=True).start()

while True:
    time.sleep(1)
