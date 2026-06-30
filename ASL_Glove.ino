#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ============================================================
//  CONFIG — EDITEAZĂ AICI
// ============================================================


const char* WIFI_SSID = "REȚEAUA_TA_WIFI";
const char* WIFI_PASS = "PAROLA_TA_WIFI";


const char* SERVER_HOST = "";
const uint16_t SERVER_PORT = 8000;
const char* SERVER_PATH = "/ws/glove";

// ============================================================
//  PIN MAP & TUNABILE
// ============================================================

// I2C: LilyGO T7 ESP32-S3 — GPIO 8 = SDA, GPIO 9 = SCL.
#define I2C_SDA 8
#define I2C_SCL 9

// Câte pachete de date trimitem pe secundă. 
const uint16_t SAMPLE_RATE_HZ = 30;
const uint32_t SAMPLE_PERIOD_MS = 1000UL / SAMPLE_RATE_HZ;


// ============================================================
//  OBIECTE HARDWARE
// ============================================================

Adafruit_ADS1115 ads1;   // adresa 0x48 — canalele 0..3 pentru flex 0..3
Adafruit_ADS1115 ads2;   // adresa 0x49 — canalul 0 pentru flex 4
Adafruit_MPU6050 mpu;    // adresa 0x68

WebSocketsClient ws;

bool ads1Ok = false;
bool ads2Ok = false;
bool mpuOk  = false;
bool wsConnected = false;

uint32_t lastSampleMs = 0;
uint32_t lastBlinkMs = 0;

// ============================================================
//  WEBSOCKET EVENT HANDLER
// ============================================================

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      wsConnected = false;
      Serial.println(F("[WS] Deconectat"));
      break;
    case WStype_CONNECTED: {
      wsConnected = true;
      Serial.printf("[WS] Conectat la %s\n", (char*)payload);
      // Trimitem un mesaj de hello ca server-ul să știe că e un glove.
      StaticJsonDocument<128> hello;
      hello["type"] = "hello";
      hello["device"] = "asl_glove_v1";
      hello["fw"] = "1.0";
      char buf[128];
      size_t n = serializeJson(hello, buf);
      ws.sendTXT(buf, n);
      break;
    }
    case WStype_TEXT:
      Serial.printf("[WS] Mesaj de la server: %s\n", (char*)payload);
      break;
    case WStype_ERROR:
      Serial.println(F("[WS] ERROR"));
      break;
    default:
      break;
  }
}

// ============================================================
//  SETUP
// ============================================================

void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println(F("\n\n[ASL Glove] Boot..."));

  // --- I2C ---
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000); // I2C fast mode

  // --- ADS1115 #1 (0x48) ---
  if (!ads1.begin(0x48, &Wire)) {
    Serial.println(F("[X] ADS1115 #1 (0x48) NEDETECTAT — verifică alimentare + SDA/SCL"));
  } else {
    ads1Ok = true;
    ads1.setGain(GAIN_ONE);
    ads1.setDataRate(RATE_ADS1115_250SPS);
    Serial.println(F("[OK] ADS1115 #1 ready"));
  }

  // --- ADS1115 #2 (0x49) ---
  if (!ads2.begin(0x49, &Wire)) {
    Serial.println(F("[X] ADS1115 #2 (0x49) NEDETECTAT — leagă ADDR la 3V3"));
  } else {
    ads2Ok = true;
    ads2.setGain(GAIN_ONE);
    ads2.setDataRate(RATE_ADS1115_250SPS);
    Serial.println(F("[OK] ADS1115 #2 ready"));
  }

  // --- MPU6050 (0x68) ---
  if (!mpu.begin(0x68, &Wire)) {
    Serial.println(F("[X] MPU6050 (0x68) NEDETECTAT — verifică SDA/SCL"));
  } else {
    mpuOk = true;
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    Serial.println(F("[OK] MPU6050 ready"));
  }

  // --- WiFi ---
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("[WiFi] Conectare la \"%s\"", WIFI_SSID);
  uint8_t tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 40) {
    delay(500);
    Serial.print('.');
    tries++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] OK — IP local: %s\n",
                  WiFi.localIP().toString().c_str());
  } else {
    Serial.println(F("\n[WiFi] EȘUAT — verifică SSID/parola; restart..."));
    ESP.restart();
  }

  // --- WebSocket client ---
  Serial.printf("[WS] Țintă: ws://%s:%u%s\n",
                SERVER_HOST, SERVER_PORT, SERVER_PATH);
  ws.begin(SERVER_HOST, SERVER_PORT, SERVER_PATH);
  ws.onEvent(webSocketEvent);
  ws.setReconnectInterval(2000);

  Serial.println(F("[Setup] Gata. Trec în loop()."));
}

// ============================================================
//  LOOP
// ============================================================

void loop() {
  ws.loop();

  uint32_t now = millis();


  if (now - lastBlinkMs > 5000) {
    lastBlinkMs = now;
    Serial.printf("[hb] WiFi:%s WS:%s ADS1:%s ADS2:%s MPU:%s\n",
                  WiFi.status() == WL_CONNECTED ? "OK" : "--",
                  wsConnected ? "OK" : "--",
                  ads1Ok ? "OK" : "--",
                  ads2Ok ? "OK" : "--",
                  mpuOk  ? "OK" : "--");
  }

  if (now - lastSampleMs < SAMPLE_PERIOD_MS) return;
  lastSampleMs = now;

  // --- Citim toate canalele ---
  float flex_v[5] = {0, 0, 0, 0, 0};
  if (ads1Ok) {
    for (int ch = 0; ch < 4; ch++) {
      int16_t raw = ads1.readADC_SingleEnded(ch);
      flex_v[ch] = ads1.computeVolts(raw);
    }
  }
  if (ads2Ok) {
    int16_t raw = ads2.readADC_SingleEnded(0);
    flex_v[4] = ads2.computeVolts(raw);
  }

  float ax = 0, ay = 0, az = 0;
  float gx = 0, gy = 0, gz = 0;
  if (mpuOk) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    ax = a.acceleration.x;   ay = a.acceleration.y;   az = a.acceleration.z;
    gx = g.gyro.x;           gy = g.gyro.y;           gz = g.gyro.z;
  }

  // --- Construim JSON-ul de trimis ---
  StaticJsonDocument<320> doc;
  doc["type"] = "glove_frame";
  doc["t"] = now;
  JsonArray jf = doc.createNestedArray("flex");
  for (int i = 0; i < 5; i++) jf.add(round(flex_v[i] * 1000) / 1000.0);
  JsonArray ja = doc.createNestedArray("accel");
  ja.add(ax); ja.add(ay); ja.add(az);
  JsonArray jg = doc.createNestedArray("gyro");
  jg.add(gx); jg.add(gy); jg.add(gz);

  if (wsConnected) {
    char buf[320];
    size_t n = serializeJson(doc, buf);
    ws.sendTXT(buf, n);
  }

  // Debug pe Serial o dată pe secundă
  static uint32_t lastDbg = 0;
  if (now - lastDbg > 1000) {
    lastDbg = now;
    Serial.printf("flex: %.3f %.3f %.3f %.3f %.3f | "
                  "accel: %.2f %.2f %.2f | gyro: %.2f %.2f %.2f | %s\n",
                  flex_v[0], flex_v[1], flex_v[2], flex_v[3], flex_v[4],
                  ax, ay, az, gx, gy, gz,
                  wsConnected ? "WS:OK" : "WS:--");
  }
}
