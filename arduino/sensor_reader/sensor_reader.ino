const int RAW_PINS[4] = {A0, A2, A4, A6};
const int PROCESSED_PINS[4] = {A1, A3, A5, A7};

void setup() {
  Serial.begin(115200);
}

void loop() {
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
  delay(10);
}
