#include <Servo.h>

const int RAW_PINS[4] = {A0, A2, A4, A6};
const int PROCESSED_PINS[4] = {A1, A3, A5, A7};
const int MOTOR_PINS[6] = {11, 10, 9, 6, 5, 3};

Servo servos[6];

// Servo pulse width ranges (microseconds)
// Adjust these for each servo type to get proper 0-180° movement
// MG996R typically: 1000-2000us, but can often go 500-2500us
// SG90 typically: 500-2500us for full range
const int SERVO_MIN_US[6] = {500, 500, 500, 500, 500, 500};  // Min pulse width
const int SERVO_MAX_US[6] = {2500, 2500, 2500, 2500, 2500, 1400};  // Servo 5 max reduced to 700

// Servo 6 is a SG90

// Max input values (0-1024 scale)
// Use 1024 for full range within the pulse width limits above
const int SERVO_MAX_INPUT[6] = {1024, 1024, 1024, 900, 1024, 1024};  // Servo 5 now uses pulse width limit

void setup() {
  Serial.begin(115200);

  // Attach servos to pins with custom pulse width range
  for (int i = 0; i < 6; i++) {
    servos[i].attach(MOTOR_PINS[i], SERVO_MIN_US[i], SERVO_MAX_US[i]);
    // Start at minimum position (0)
    servos[i].writeMicroseconds(SERVO_MIN_US[i]);
  }

}

void loop() {

  // Read sensor data
  for (int i = 0; i < 4; i++) {
    int raw = analogRead(RAW_PINS[i]);
    int env = analogRead(PROCESSED_PINS[i]);
    Serial.print("S");
    Serial.print(4 - i);
    Serial.print(":");
    Serial.print(raw);
    Serial.print(",");
    Serial.print(env);
    if (i < 3) Serial.print(";");
  }
  Serial.println();

  // Check for servo control commands
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    int servoValues[6];
    int valueIndex = 0;
    int startPos = 0;

    for (int i = 0; i <= input.length() && valueIndex < 6; i++) {
      if (i == input.length() || input.charAt(i) == ',') {
        String valueStr = input.substring(startPos, i);
        int value = valueStr.toInt();

        // Map from 0-1024 (input) to 0-SERVO_MAX_INPUT (servo's limit)
        // Then map to pulse width range (microseconds)
        int mappedInput = map(constrain(value, 0, 1024), 0, 1024, 0, SERVO_MAX_INPUT[valueIndex]);
        servoValues[valueIndex] = map(mappedInput, 0, SERVO_MAX_INPUT[valueIndex],
                                      SERVO_MIN_US[valueIndex], SERVO_MAX_US[valueIndex]);
        valueIndex++;
        startPos = i + 1;
      }
    }

    // Apply values to servos if we got 6 values
    if (valueIndex == 6) {
      for (int i = 0; i < 6; i++) {
        servos[i].writeMicroseconds(servoValues[i]);
      }
    }
  }

  delay(33);  // ~30Hz sampling rate (matches data collection)
}
