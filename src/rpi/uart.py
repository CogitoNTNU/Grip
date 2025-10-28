import threading
import time

import serial

# SAME51 Curiosity Nano på macOS
ser = serial.Serial("/dev/cu.usbmodem1202", 115200, timeout=0.5)
time.sleep(2)

last_number = None
lock = threading.Lock()


def reader():
    global last_number
    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline()
                if line:
                    msg = line.decode("utf-8", errors="ignore").strip()
                    print("Mottatt:", msg)
                    try:
                        number = int(msg)
                        with lock:
                            last_number = number
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Read error: {e}")
        time.sleep(0.01)


def writer():
    global last_number
    while True:
        try:
            with lock:
                if last_number is not None:
                    echo = f"{last_number}\n".encode("utf-8")
                    ser.write(echo)
                    print("Sendt tilbake:", last_number)
        except Exception as e:
            print(f"Write error: {e}")
        time.sleep(2)  # send hvert 2. sekund


print("Kommunikasjon startet med /dev/cu.usbmodem1302 @ 115200 baud")
print("Trykk Ctrl+C for å avslutte.\n")

threading.Thread(target=reader, daemon=True).start()
threading.Thread(target=writer, daemon=True).start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nAvslutter...")
    ser.close()
