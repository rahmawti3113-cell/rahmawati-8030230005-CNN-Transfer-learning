/*
 * esp32_relay_link.ino
 * 4-channel relay driver for ESP32, controlled over USB serial.
 *
 * Line protocol @ 115200 baud (newline-terminated):
 *     "n1" -> relay n ON     "n0" -> relay n OFF   (n = 1..4)
 *     "A1" -> all ON         "A0" -> all OFF
 *   e.g. 31 = relay 3 on, 30 = relay 3 off
 *
 * Relay inputs IN1..IN4 wired to GPIO 23, 19, 18, 5.
 * Relay coils on 5V (VIN), grounds common. Most boards are active-LOW;
 * set ACTIVE_LOW=false if yours triggers on HIGH.
 */

#define CHANNELS   4
#define ACTIVE_LOW true

const uint8_t PINS[CHANNELS] = {5, 19, 18, 23};
bool          isOn[CHANNELS] = {false, false, false, false};

// translate a logical state into the pin level this board expects
static inline uint8_t toLevel(bool on) {
  return (ACTIVE_LOW ? (on ? LOW : HIGH) : (on ? HIGH : LOW));
}

static void writeChannel(uint8_t ch, bool on) {
  if (ch >= CHANNELS) return;
  isOn[ch] = on;
  digitalWrite(PINS[ch], toLevel(on));
  Serial.print("relay ");
  Serial.print(ch + 1);
  Serial.println(on ? " ON" : " OFF");
}

static void writeAll(bool on) {
  for (uint8_t i = 0; i < CHANNELS; i++) {
    isOn[i] = on;
    digitalWrite(PINS[i], toLevel(on));
  }
  Serial.println(on ? "all ON" : "all OFF");
}

// parse one line of the form <target><state>
static void dispatch(const String &raw) {
  String s = raw;
  s.trim();
  if (s.length() < 2) return;

  char target = s[0];
  bool on     = (s[1] == '1');

  if (target == 'A' || target == 'a') {
    writeAll(on);
  } else if (target >= '1' && target <= '4') {
    writeChannel(target - '1', on);
  } else {
    Serial.print("ignored: ");
    Serial.println(s);
  }
}

void setup() {
  Serial.begin(115200);
  delay(300);

  for (uint8_t i = 0; i < CHANNELS; i++) {
    digitalWrite(PINS[i], toLevel(false));   // keep off through init
    pinMode(PINS[i], OUTPUT);
    digitalWrite(PINS[i], toLevel(false));
  }

  Serial.println();
  Serial.println("relay link up | n1/n0 (n=1..4), A1/A0");
}

void loop() {
  if (Serial.available()) {
    dispatch(Serial.readStringUntil('\n'));
  }
}
