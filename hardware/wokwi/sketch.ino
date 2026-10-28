/*
  SightSeeker rover: Wokwi simulation build
  -----------------------------------------
  The SAME firmware as hardware/sightseeker_rover/sightseeker_rover.ino (serial protocol, 500 ms
  watchdog, ultrasonic emergency stop), plus two things that only exist for the simulation:

  1. Motor stand-ins. Wokwi has no L298N or DC motor parts, so each motor is shown with LEDs wired to the
     exact pins the L298N would use: green = IN1/IN3 (forward), red = IN2/IN4 (reverse), yellow = ENA/ENB
     (PWM speed, brightness = speed). An OLED shows the wheel speeds, distance and safety state.
  2. Demo mode. If no command arrives on the serial port for 3 s, the sketch replays a recorded mission
     from the Vision Brain at the same 20 commands/s it sends in real life: scan, turn to the target,
     drive, avoid an obstacle, arrive, then a "lost link" to show the watchdog stopping the motors.

  Try it:  drag the HC-SR04 distance below 25 cm during a "forward" step -> forward motion is blocked.
           Type a command in the Serial Monitor, e.g.  L150,R150  (motors stop 0.5 s later: watchdog).
*/
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

const int ENA = 5, IN1 = 7, IN2 = 8;    // left motor  (L298N)
const int ENB = 6, IN3 = 9, IN4 = 10;   // right motor (L298N)
const int TRIG = 11, ECHO = 12;         // HC-SR04
const unsigned long WATCHDOG_MS = 500;
const int STOP_CM = 25;

Adafruit_SSD1306 oled(128, 64, &Wire, -1);
bool oledOk = false;

int cmdLeft = 0, cmdRight = 0;          // last command received
int outLeft = 0, outRight = 0;          // what the motors actually get
long distCm = 999;
bool blocked = false, watchdogStop = true;
unsigned long lastPing = 0, lastCmd = 0, lastDraw = 0, lastSerial = 0;
char buf[32];
byte len = 0;

void setMotor(int en, int inA, int inB, int speed) {
  speed = constrain(speed, -255, 255);
  digitalWrite(inA, speed > 0 ? HIGH : LOW);
  digitalWrite(inB, speed < 0 ? HIGH : LOW);
  analogWrite(en, abs(speed));
}

void applyCommand() {
  int l = cmdLeft, r = cmdRight;
  blocked = (distCm < STOP_CM && l > 0 && r > 0);      // never drive forward into something close
  if (blocked) { l = 0; r = 0; }
  if (watchdogStop) { l = 0; r = 0; }
  outLeft = l; outRight = r;
  setMotor(ENA, IN1, IN2, l);
  setMotor(ENB, IN3, IN4, r);
}

long readDistanceCm() {
  digitalWrite(TRIG, LOW); delayMicroseconds(2);
  digitalWrite(TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  unsigned long us = pulseIn(ECHO, HIGH, 25000UL);
  return us == 0 ? 999 : (long)(us / 58);
}

bool parseCommand(const char *s, int &left, int &right) {
  if (s[0] != 'L') return false;
  const char *comma = strchr(s, ',');
  if (!comma || comma[1] != 'R') return false;
  left = atoi(s + 1);
  right = atoi(comma + 2);
  return true;
}

void handleLine(const char *line) {           // same path for real serial input and the demo
  int l, r;
  if (parseCommand(line, l, r)) {
    cmdLeft = l; cmdRight = r;
    lastCmd = millis();
    watchdogStop = false;
    applyCommand();
  }
}

// ---------------- demo mode (simulation only) ----------------
struct Step { int l, r; unsigned int ms; bool send; const char *label; };
const Step DEMO[] = {
  {  90,  -90, 3000, true,  "SEARCH: scan right" },
  { 110,  200, 2000, true,  "TARGET: turn left" },
  { 180,  180, 3000, true,  "NAVIGATE: forward" },
  {  40,  210, 2000, true,  "AVOID: obstacle->left" },
  { 170,  180, 2500, true,  "NAVIGATE: forward" },
  {   0,    0, 2500, true,  "GOAL REACHED: stop" },
  { 150,  150, 1500, true,  "NAVIGATE: forward" },
  { 150,  150, 2500, false, "LINK LOST: no cmds" },
};
const int DEMO_N = sizeof(DEMO) / sizeof(DEMO[0]);
int demoStep = -1;
unsigned long demoStart = 0, demoLastSend = 0;
const char *label = "waiting";

void runDemo() {
  unsigned long now = millis();
  if (now - lastSerial < 3000) { demoStep = -1; label = "SERIAL: laptop"; return; }   // real commands win
  if (demoStep < 0 || now - demoStart >= DEMO[demoStep].ms) {
    demoStep = (demoStep + 1) % DEMO_N;
    demoStart = now;
    label = DEMO[demoStep].label;
    Serial.print(F("[demo] ")); Serial.println(label);
  }
  if (DEMO[demoStep].send && now - demoLastSend >= 50) {          // 20 commands per second
    demoLastSend = now;
    char line[24];
    snprintf(line, sizeof(line), "L%d,R%d", DEMO[demoStep].l, DEMO[demoStep].r);
    handleLine(line);
  }
}

void drawBar(int y, const char *name, int v) {
  oled.setCursor(0, y); oled.print(name); oled.print(v >= 0 ? "+" : ""); oled.print(v);
  int w = abs(v) * 40 / 255;
  oled.drawRect(64, y, 62, 7, SSD1306_WHITE);
  if (v >= 0) oled.fillRect(95, y + 1, w * 30 / 40, 5, SSD1306_WHITE);
  else oled.fillRect(95 - w * 30 / 40, y + 1, w * 30 / 40, 5, SSD1306_WHITE);
}

void drawOled() {
  if (!oledOk) return;
  oled.clearDisplay();
  oled.setTextSize(1); oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(0, 0); oled.print(F("SIGHTSEEKER ROVER"));
  oled.setCursor(0, 10); oled.print(label);
  drawBar(22, "L ", outLeft);
  drawBar(32, "R ", outRight);
  oled.setCursor(0, 44); oled.print(F("dist ")); if (distCm >= 999) oled.print(F("--")); else oled.print(distCm); oled.print(F(" cm"));
  oled.setCursor(0, 55);
  if (watchdogStop) oled.print(F("WATCHDOG: STOPPED"));
  else if (blocked) oled.print(F("BLOCKED: too close"));
  else oled.print(F("OK"));
  oled.display();
}

void setup() {
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);
  Serial.begin(115200);
  oledOk = oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  applyCommand();
  Serial.println(F("SightSeeker rover sim. Send L<left>,R<right> (-255..255), or wait 3 s for the demo."));
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    lastSerial = millis();
    if (c == '\n' || c == '\r') {
      if (len > 0) { buf[len] = '\0'; handleLine(buf); len = 0; }
    } else if (len < sizeof(buf) - 1) {
      buf[len++] = c;
    } else {
      len = 0;
    }
  }
  runDemo();
  if (millis() - lastPing > 60) {
    lastPing = millis();
    distCm = readDistanceCm();
    applyCommand();
  }
  if (!watchdogStop && millis() - lastCmd > WATCHDOG_MS) {
    watchdogStop = true; cmdLeft = cmdRight = 0;
    applyCommand();
    Serial.println(F("[safety] no command for 500 ms -> motors stopped"));
  }
  if (millis() - lastDraw > 100) { lastDraw = millis(); drawOled(); }
}
