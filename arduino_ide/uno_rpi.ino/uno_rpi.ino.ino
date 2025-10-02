const int SENSOR_PIN = A0;
const int FULL_SCALE = 1023; // bruk 4095 på 12-bit ADC (f.eks. ESP32)

void setup() {
  Serial.begin(9600);
}

void loop() {
  int sensorValue = analogRead(SENSOR_PIN);
  Serial.println(sensorValue); // bare send råverdien
  Serial.print('\t');
  Serial.println(FULL_SCALE);
  delay(50);
}
