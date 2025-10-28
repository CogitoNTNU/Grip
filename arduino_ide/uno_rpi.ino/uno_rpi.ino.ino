int analogPin = A0;       // Inngang
int pwmPins[] = {3, 5, 6, 9}; // Utganger (PWM-støtte)
int pwmValue = 0;

void setup() {
  Serial.begin(9600); // Start seriell kommunikasjon
  for (int i = 0; i < 4; i++) {
    pinMode(pwmPins[i], OUTPUT);
  }
}

void loop() {
  int sensorValue = analogRead(analogPin);       // 0–1023
  pwmValue = map(sensorValue, 0, 1023, 0, 255); // skaler til 0–255

  // Skriv til seriell monitor
  Serial.print("Analog verdi: ");
  Serial.print(sensorValue);
  Serial.print("  PWM-verdi: ");
  Serial.println(pwmValue);

  for (int i = 0; i < 4; i++) {
    analogWrite(pwmPins[i], pwmValue);          // skriv samme PWM ut
  }

  delay(100); // litt delay for å lese enklere i monitor
}