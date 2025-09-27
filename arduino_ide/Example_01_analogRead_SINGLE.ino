/*
  MyoWare Example_01_analogRead_SINGLE
  SparkFun Electronics
  Pete Lewis
  3/24/2022
  License: This code is public domain but you buy me a beverage if you use this and we meet someday.
  This code was adapted from the MyoWare analogReadValue.ino example found here:
  https://github.com/AdvancerTechnologies/MyoWare_MuscleSensor

  This example streams the data from a single MyoWare sensor attached to ADC A0.
  Graphical representation is available using Serial Plotter (Tools > Serial Plotter menu).

  *Only run on a laptop using its battery. Do not plug in laptop charger/dock/monitor.
  
  *Do not touch your laptop trackpad or keyboard while the MyoWare sensor is powered.

  Hardware:
  SparkFun RedBoard Artemis (or Arduino of choice)
  USB from Artemis to Computer.
  Output from sensor connected to your Arduino pin A0
  
  This example code is in the public domain.
*/

const int SENSOR_PIN = A0;
const int FULL_SCALE = 1023; // bruk 4095 på 12-bit ADC (f.eks. ESP32)

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  Serial.println("MyoWare dual print");
}

void loop() {
  int v = analogRead(A0);
  Serial.print(0);
  Serial.print('\t');
  Serial.print(v);
  Serial.print('\t');
  Serial.println(1023);
}
