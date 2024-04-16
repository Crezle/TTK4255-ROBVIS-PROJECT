#include <WebServer.h>
#include <WiFi.h>
#include <esp32cam.h>
#include <ArduinoJson.h>
#include <fstream>
#include "esp_camera.h"
#include <HTTPClient.h>
 
const char* WIFI_SSID = "Galaxy S10e";
const char* WIFI_PASS = "Juve51234";
const int Direction1 = 1;
const int Direction2 = 2;
const int greenLight1 = GPIO_NUM_12;
const int yellowLight1 = GPIO_NUM_13;
const int redLight1 = GPIO_NUM_15;
const int greenLight2 = GPIO_NUM_14;
const int yellowLight2 = GPIO_NUM_2;
const int redLight2 = GPIO_NUM_4;
int carsDirection1 = 0;
int carsDirection2 = 0;
 
WebServer server(80);
 
static auto loRes = esp32cam::Resolution::find(320, 240);
static auto midRes = esp32cam::Resolution::find(800, 600);
static auto hiRes = esp32cam::Resolution::find(1280, 720);
void serveJpg()
{
  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("CAPTURE FAIL");
    server.send(503, "", "");
    return;
  }
  //Serial.printf("CAPTURE OK %dx%d %db\n", frame->getWidth(), frame->getHeight(),
  //              static_cast<int>(frame->size()));
 
  server.setContentLength(frame->size());
  server.send(200, "image/jpeg");
  WiFiClient client = server.client();
  frame->writeTo(client);
}
 
void handleJpgLo()
{
  if (!esp32cam::Camera.changeResolution(loRes)) {
    Serial.println("SET-LO-RES FAIL");
  }
  serveJpg();
}
 
void handleJpgHi()
{
  if (!esp32cam::Camera.changeResolution(hiRes)) {
    Serial.println("SET-HI-RES FAIL");
  }
  serveJpg();
}
 
void handleJpgMid()
{
  if (!esp32cam::Camera.changeResolution(midRes)) {
    Serial.println("SET-MID-RES FAIL");
  }
  serveJpg();
}

void turnOnLights(int direction)
{
  if (direction == Direction1) {
    // Turn on green light in direction 1
    digitalWrite(redLight1, LOW);
    Serial.println("Green light 1 is on");
    digitalWrite(greenLight1, HIGH);
  } else {
    // Turn on green lights in direction 2
    digitalWrite(redLight2, LOW);
    Serial.println("Green light 2 is on");
    digitalWrite(greenLight2, HIGH);
  }
}
void turnOffLights(int direction)
{
  if (direction == Direction1) {
    // Turn off green light in direction 1
    digitalWrite(greenLight1, LOW);
    Serial.println("Yellow light 1 is on");
    digitalWrite(yellowLight1, HIGH);
    delay(1000);
    digitalWrite(yellowLight1, LOW);
    Serial.println("Red light 1 is on");
    digitalWrite(redLight1, HIGH);
  } else {
    // Turn off green lights in direction 2
    digitalWrite(greenLight2, LOW);
    Serial.println("Yellow light 2 is on");
    digitalWrite(yellowLight2, HIGH);
    delay(1000);
    digitalWrite(yellowLight2, LOW);
    Serial.println("Red light 2 is on");
    digitalWrite(redLight2, HIGH);
  }
}

void handeTraficLights()
{
HTTPClient http;
http.setTimeout(5000);
String url = "https://raw.githubusercontent.com/Crezle/TTK4255-ROBVIS-PROJECT/main/data/data.json";

while (true) {
    Serial.println("URL: " + url);
    http.begin(url);
    int httpCode = http.GET();
    Serial.println(httpCode);
    DynamicJsonDocument doc(1024);
    if (httpCode > 0) {
        if (httpCode == HTTP_CODE_OK) {
            String payload = http.getString();
            Serial.println("Payload: " + payload);
            DeserializationError error = deserializeJson(doc, payload);
            if (error) {
                Serial.print("Parsing failed: ");
                Serial.println(error.c_str());
                return;
            }
            carsDirection1 = doc["direction1"];
            carsDirection2 = doc["direction2"];
            Serial.print("Direction 1: ");
            Serial.println(carsDirection1);
            Serial.print("Direction 2: ");
            Serial.println(carsDirection2);
            break; 
        } else if (httpCode == HTTP_CODE_MOVED_PERMANENTLY || httpCode == HTTP_CODE_FOUND) {
            // Handle redirection
            url = http.header("Location"); 
            if (url.startsWith("http://") || url.startsWith("https://")) {
                Serial.print("Redirecting to: ");
                Serial.println(url);
                http.end(); 
            } else {
                Serial.println("Invalid URL: " + url);
                break; 
            }
        } else {
            Serial.println("Error on HTTP code");
            break; 
        }
    } else {
        Serial.println("Error on HTTP request");
        break; // Exit loop on error
    }
}

http.end();

 if (carsDirection1 == carsDirection2) {
    // Turn on lights for 10 seconds in one direction, then turn off
    turnOnLights(Direction1);
    delay(10000);
    turnOffLights(Direction1);

    // Turn on lights for 10 seconds in the other direction
    turnOnLights(Direction2);
    delay(10000);
    turnOffLights(Direction2);
} else if (carsDirection1 > 2 * carsDirection2) {
    // Turn on lights in the direction with more cars with the duration twice the one in the other direction
    turnOnLights(Direction1);
    delay(20000);
    turnOffLights(Direction1);

    // Turn on lights in the other direction for half the time
    turnOnLights(Direction2);
    delay(10000);
    turnOffLights(Direction2);
    delay(10000);
} else if (carsDirection2 > 2 * carsDirection1) {
    // Turn on lights in the direction with more cars with the duration twice the one in the other direction
    turnOnLights(Direction2);
    delay(20000);
    turnOffLights(Direction2);

    // Turn on lights in the other direction for half the time
    turnOnLights(Direction1);
    delay(10000);
    turnOffLights(Direction1);
}
}
void handleTrafficLights(void * parameter) {
  while (true) {
    handeTraficLights();
    delay(200000);//dummy delay
  }
}

void handleClients(void * parameter) {
  while (true) {
    // Handle client requests
    server.handleClient();
  }
}
void  setup(){
  Serial.begin(115200);
  Serial.println();
  {
    using namespace esp32cam;
    Config cfg;
    cfg.setPins(pins::AiThinker);
    cfg.setResolution(hiRes);
    cfg.setBufferCount(2);
    cfg.setJpeg(80);
    bool ok = Camera.begin(cfg);
    Serial.println(ok ? "CAMERA OK" : "CAMERA FAIL");
  }
  sensor_t * s = esp_camera_sensor_get();
  s->set_saturation(s, -2);     // -2 to 2
  s->set_contrast(s, 2);       // -2 to 2
  pinMode(greenLight1, OUTPUT);
  pinMode(yellowLight1, OUTPUT);
  pinMode(redLight1, OUTPUT);
  pinMode(greenLight2, OUTPUT);
  pinMode(yellowLight2, OUTPUT);
  pinMode(redLight2, OUTPUT);
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  Serial.print("http://");
  Serial.println(WiFi.localIP());
  Serial.println("  /cam-lo.jpg");
  Serial.println("  /cam-hi.jpg");
  Serial.println("  /cam-mid.jpg");
 
  server.on("/cam-lo.jpg", handleJpgLo);
  server.on("/cam-hi.jpg", handleJpgHi);
  server.on("/cam-mid.jpg", handleJpgMid);
 
  server.begin();
    // Create tasks for handling traffic lights and clients
  xTaskCreatePinnedToCore(
    handleTrafficLights,
    "TrafficLightsTask",
    10000,
    NULL,
    1,
    NULL,
    0
  );

  xTaskCreatePinnedToCore(
    handleClients,
    "ClientsTask",
    10000,
    NULL,
    1,
    NULL,
    1
  );
}
 
void loop()
{
  // Nothing to do here
}

