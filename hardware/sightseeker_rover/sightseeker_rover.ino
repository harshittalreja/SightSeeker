/*
  SightSeeker rover firmware (Arduino Uno/Nano + L298N dual motor driver)
  ----------------------------------------------------------------------
  Receives wheel commands from yolo_detection.py over USB serial:

      L<left>,R<right>\n        e.g.  "L120,R-80\n"      range -255..255

  and drives two DC motors (differential drive). Safety: if no valid command
  arrives for 500 ms the motors stop (e.g. laptop unplugged or script closed).

  Safety: an HC-SR04 ultrasonic sensor looks forward. If something is closer than
  STOP_CM, any forward motion is blocked (turning on the spot and reversing still work),
  so the rover cannot drive into an obstacle even if the laptop says "forward".

  Wiring (L298N):
    ENA -> D5 (PWM)   IN1 -> D7   IN2 -> D8     left motor on OUT1/OUT2
    ENB -> D6 (PWM)   IN3 -> D9   IN4 -> D10    right motor on OUT3/OUT4
    L298N GND must be connected to Arduino GND. Remove the ENA/ENB jumpers.
  Wiring (HC-SR04): VCC -> 5V, GND -> GND, TRIG -> D11, ECHO -> D12

  Run on the laptop:  python yolo_detection.py --serial COM5      (use your port)
  Note: written for the hackathon demo; check motor directions on your chassis and
  swap IN1/IN2 (or IN3/IN4) if a wheel spins backwards.
*/

const int ENA = 5, IN1 = 7, IN2 = 8;    // left motor
const int ENB = 6, IN3 = 9, IN4 = 10;   // right motor
const int TRIG = 11, ECHO = 12;         // HC-SR04 ultrasonic sensor
const unsigned long WATCHDOG_MS = 500;
const int STOP_CM = 25;                  // block forward motion closer than this
int cmdLeft = 0, cmdRight = 0;
long distCm = 999;
unsigned long lastPing = 0;

char buf[32];
byte len = 0;
unsigned long lastCmd = 0;

void setMotor(int en, int inA, int inB, int speed) {
  speed = constrain(speed, -255, 255);
  if (speed > 0) {
    digitalWrite(inA, HIGH); digitalWrite(inB, LOW);
  } else if (speed < 0) {
    digitalWrite(inA, LOW); digitalWrite(inB, HIGH);
  } else {
    digitalWrite(inA, LOW); digitalWrite(inB, LOW);
  }
  analogWrite(en, abs(speed));
}

void stopMotors() {
  setMotor(ENA, IN1, IN2, 0);
  setMotor(ENB, IN3, IN4, 0);
}

long readDistanceCm() {
  digitalWrite(TRIG, LOW); delayMicroseconds(2);
  digitalWrite(TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  unsigned long us = pulseIn(ECHO, HIGH, 25000UL);   // ~4 m max, never blocks > 25 ms
  return us == 0 ? 999 : (long)(us / 58);
}

// Apply the last command, unless it would drive forward into something close.
void applyCommand() {
  int l = cmdLeft, r = cmdRight;
  if (distCm < STOP_CM && l > 0 && r > 0) { l = 0; r = 0; }
  setMotor(ENA, IN1, IN2, l);
  setMotor(ENB, IN3, IN4, r);
}

// Parses "L<int>,R<int>". Returns true on success.
bool parseCommand(const char *s, int &left, int &right) {
  if (s[0] != 'L') return false;
  const char *comma = strchr(s, ',');
  if (!comma || comma[1] != 'R') return false;
  left = atoi(s + 1);
  right = atoi(comma + 2);
  return true;
}

void setup() {
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);
  stopMotors();
  Serial.begin(115200);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (len > 0) {
        buf[len] = '\0';
        int left, right;
        if (parseCommand(buf, left, right)) {
          cmdLeft = left; cmdRight = right;
          applyCommand();
          lastCmd = millis();
        }
        len = 0;
      }
    } else if (len < sizeof(buf) - 1) {
      buf[len++] = c;
    } else {
      len = 0;  // overflow: drop the line
    }
  }
  if (millis() - lastPing > 60) {          // ~16 distance readings per second
    lastPing = millis();
    distCm = readDistanceCm();
    applyCommand();
  }
  if (millis() - lastCmd > WATCHDOG_MS) { cmdLeft = cmdRight = 0; stopMotors(); }
}
