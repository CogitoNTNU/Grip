const int SENSOR_PIN = A0;
const int FULL_SCALE = 1023; // 4095 on 12-bit ADC

void setup() {
  Serial.begin(9600);
  // Optional: tell Python we're ready and what columns to expect
  Serial.println("READY");
  Serial.println("sensor\tfull_scale"); // header (one-time)
}

void loop() {
  int sensorValue = analogRead(SENSOR_PIN);
  Serial.print(sensorValue);
  Serial.print('\t');
  Serial.println(FULL_SCALE);  // <- exactly one newline per sample
  delay(50);
}
