/*
 * 8-Channel Sensor Reader for Grip Data Collection
 *
 * Reads 8 analog sensors (A0-A7) and sends data via Serial
 * Format: env0,raw0,env1,raw1,env2,raw2,env3,raw3
 *
 * Compatible with data_collection/collectors/integrated_collector.py
 */

const int NUM_SENSORS = 4;  // 4 sensor pairs
const int SENSOR_PINS[NUM_SENSORS] = {A0, A1, A2, A3};
const int BAUD_RATE = 115200;  // Fast baud rate for high sampling
const int SAMPLE_DELAY = 10;   // 10ms = ~100Hz sampling

void setup() {
  Serial.begin(BAUD_RATE);

  // Wait for serial connection
  while (!Serial) {
    ; // Wait for serial port to connect
  }

  // Print startup message (for debugging in Arduino IDE)
  Serial.println("# Grip Sensor System - 8 Channel");
  Serial.println("# Format: env0,raw0,env1,raw1,env2,raw2,env3,raw3");
  Serial.println("# Sampling at ~100Hz");
  delay(1000);
}

void loop() {
  // Read all sensors
  int sensorValues[NUM_SENSORS];
  for (int i = 0; i < NUM_SENSORS; i++) {
    sensorValues[i] = analogRead(SENSOR_PINS[i]);
  }

  // Send data in CSV format: env0,raw0,env1,raw1,...
  // For now, we send the same value as both "envelope" and "raw"
  for (int i = 0; i < NUM_SENSORS; i++) {
    Serial.print(sensorValues[i]);  // envelope
    Serial.print(",");
    Serial.print(sensorValues[i]);  // raw (same as envelope for now)

    if (i < NUM_SENSORS - 1) {
      Serial.print(",");
    }
  }
  Serial.println();  // End of line

  delay(SAMPLE_DELAY);
}
