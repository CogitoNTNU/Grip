# UART Communication between Arduino UNO and Raspberry Pi 5

This README describes how to set up a simple UART link between an Arduino UNO
and a Raspberry Pi 5, for streaming analog sensor values (e.g. MyoWare muscle sensor).

---

## Hardware Connections

- **Arduino UNO TX (D1)** → **Voltage divider** → **Raspberry Pi GPIO15 (RXD, physical pin 10)**
- **Arduino GND** → **Raspberry Pi GND (physical pin 6)**
- Do **not** connect Arduino RX unless you also want to send data back from the Pi.

### Voltage Divider
Since Arduino UNO TX is 5V and the Pi only accepts 3.3V:
- R1 = 1.8 kΩ (between UNO TX and divider node)
- R2 = 3.3 kΩ (between divider node and GND)
- Divider node → Pi RX

This scales 5V down to ~3.3V.

---

## Raspberry Pi Configuration

1. Enable the UART:
   ```bash
   sudo raspi-config
   ```
   - Disable login shell over serial
   - Enable serial port hardware
   - Reboot

2. The Pi 5 UART is available at `/dev/ttyAMA0`.

---

## Testing the Connection

### Check wiring and stream at 9600 baud
On Raspberry Pi:
```bash
stty -F /dev/ttyAMA0 9600 cs8 -cstopb -parenb raw -echo -ixon -ixoff -crtscts
cat /dev/ttyAMA0
```

You should see text like `HELLO 1`, `HELLO 2`, etc. from the Arduino test script.

### Using higher baud rates
You can increase to 57600 or 115200 once wiring is stable:
```bash
stty -F /dev/ttyAMA0 57600 cs8 -cstopb -parenb raw -echo -ixon -ixoff -crtscts
cat /dev/ttyAMA0
```

If you see gibberish, lower the baud rate (9600 is always safe).

---

## Python Reader

This repository includes `uart.py` in the `src/` folder.  
It automatically opens `/dev/ttyAMA0` at the configured baud (default 9600) and prints each line.

Run it with:
```bash
python uart.py
```

---

## Workflow Summary

1. Upload the Arduino sketch from `arduino_ide/` to your UNO.
2. Connect TX→RX via voltage divider and GND↔GND.
3. On the Raspberry Pi:
   - Configure the UART with `stty`
   - Run `cat /dev/ttyAMA0` to verify data
   - Run `python uart.py` to stream data into Python

---

## Notes

- Keep wires short and GNDs connected.
- If using other Arduino boards (Leonardo, Mega, ESP32), the hardware serial port name may differ (`Serial1`, etc.).
- Baud rate must match on both Arduino and Pi.
