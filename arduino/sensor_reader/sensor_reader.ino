/*
 * Grip Sensor Reader - 8 Channel (4 Sensors)
 *
 * Reads analog signals from 4 sensors with raw and processed data
 *
 * Pin Mapping (Sensors numbered 4, 3, 2, 1):
 * - A0: Raw data from Sensor 4
 * - A1: Processed data from Sensor 4
 * - A2: Raw data from Sensor 3
 * - A3: Processed data from Sensor 3
 * - A4: Raw data from Sensor 2
 * - A5: Processed data from Sensor 2
 * - A6: Raw data from Sensor 1
 * - A7: Processed data from Sensor 1
 *
 * Serial Output Format (CSV):
 * env0,raw0,env1,raw1,env2,raw2,env3,raw3
 *
 * Where:
 * - env0, raw0 = Sensor 4 (processed, raw)
 * - env1, raw1 = Sensor 3 (processed, raw)
 * - env2, raw2 = Sensor 2 (processed, raw)
 * - env3, raw3 = Sensor 1 (processed, raw)
 */

const int BAUD_RATE = 115200;
const int SAMPLE_DELAY = 10;  // 10ms = ~100Hz

// Pin definitions
const int RAW_PINS[4] = {A0, A2, A4, A6};       // Raw data: Sensor 4, 3, 2, 1
const int PROCESSED_PINS[4] = {A1, A3, A5, A7}; // Processed data: Sensor 4, 3, 2, 1

void setup() {
  Serial.begin(BAUD_RATE);

  // Wait for serial connection
  while (!Serial) {
    delay(10);
  }

  // Print header info (for Arduino Serial Monitor)
  Serial.println("# Grip Sensor System - 4 Sensors, 8 Channels");
  Serial.println("# Sensor Order: 4, 3, 2, 1");
  Serial.println("# Format: env0,raw0,env1,raw1,env2,raw2,env3,raw3");
  Serial.println("# Sampling at ~100Hz");
  Serial.println("#");
  delay(1000);
}

void loop() {
  // Read all 8 channels
  int raw[4];
  int processed[4];

  for (int i = 0; i < 4; i++) {
    processed[i] = analogRead(PROCESSED_PINS[i]);  // Envelope/processed
    raw[i] = analogRead(RAW_PINS[i]);              // Raw sensor data
  }

  // Send as CSV: env0,raw0,env1,raw1,env2,raw2,env3,raw3
  // (processed,raw pairs for sensors 4,3,2,1)
  for (int i = 0; i < 4; i++) {
    Serial.print(processed[i]);
    Serial.print(",");
    Serial.print(raw[i]);

    if (i < 3) {
      Serial.print(",");
    }
  }
  Serial.println();

  delay(SAMPLE_DELAY);
}
