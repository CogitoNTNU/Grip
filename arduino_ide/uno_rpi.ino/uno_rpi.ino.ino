void setup() {
  Serial.begin(57600);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  static unsigned long i = 0;
  Serial.print("HELLO ");
  Serial.println(i++);
  digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
  delay(100);
}