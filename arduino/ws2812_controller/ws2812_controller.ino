
/*
  This example works for ESP8266 & ESP32 and uses the NeoPixelBus library instead of the one bundle
  Sketch written by Joey Babcock - https://joeybabcock.me/blog/, and Scott Lawson (Below)
  Codebase created by ScottLawsonBC - https://github.com/scottlawsonbc
*/

#include <Adafruit_NeoPixel.h>
#include <WiFiNINA.h>        // Wifi library
#include <WiFiUdp.h>         // UDP library
#include <Arduino_LSM6DS3.h> // IMU accelerometer library

// Set to the number of LEDs in your LED strip
#define NUM_LEDS 256
// Maximum number of packets to hold in the buffer. Don't change this.
#define BUFFER_LEN 1024
// Toggles FPS output (1 = print FPS over serial, 0 = disable output)
#define PRINT_FPS 0

// NeoPixelBus settings
const uint8_t PixelPin = 3; // make sure to set this to the correct pin, ignored for Esp8266(set to 3 by default for DMA)

// Wifi and socket settings
const char *ssid = "TobinGuest";
const char *password = "cole&christian";
unsigned int localPort = 7777;
char packetBuffer[BUFFER_LEN];

uint8_t N = 0;

int status = WL_IDLE_STATUS; // Status of WiFi connection

WiFiSSLClient client; // Instantiate the Wifi client

WiFiUDP port;
// Network information
// IP must match the IP in config.py
IPAddress ip(192, 168, 0, 150);
// Set gateway to your router's gateway
IPAddress gateway(192, 168, 0, 1);
IPAddress subnet(255, 255, 255, 0);
Adafruit_NeoPixel ledstrip(NUM_LEDS, PixelPin, NEO_GRB + NEO_KHZ800);

void setup()
{

  // Output GCLK4 at 16MHz on PA20 (Nano 33 IoT digital pin D9)

  GCLK->GENDIV.reg = GCLK_GENDIV_DIV(3) | // Divide the 48MHz clock source by divisor 1: 48MHz/3=16MHz
                     GCLK_GENDIV_ID(4);   // Select GCLK4

  GCLK->GENCTRL.reg = GCLK_GENCTRL_OE |          // Enable GCLK4 output
                      GCLK_GENCTRL_IDC |         // Set the duty cycle to 50/50 HIGH/LOW
                      GCLK_GENCTRL_GENEN |       // Enable GCLK3
                      GCLK_GENCTRL_SRC_DFLL48M | // Set the 48MHz DFLL48M clock source
                      GCLK_GENCTRL_ID(4);        // Select GCLK4's ID

  PORT->Group[PORTA].PINCFG[20].bit.PMUXEN = 1;              // Switch on the port multiplexer on PA20
  PORT->Group[PORTA].PMUX[20 >> 1].reg |= PORT_PMUX_PMUXE_H; // Activate the GCLK_IO[4] on this pin

  Serial.begin(115200);
  //  WiFi.mode(WIFI_STA);
  WiFi.config(ip, gateway, subnet);
  WiFi.begin(ssid, password);
  Serial.println("");
  // Connect to wifi and print the IP address over serial
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.print("Connected to ");
  Serial.println(ssid);
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  port.begin(localPort);

  ledstrip.clear();
  ledstrip.begin(); // Begin output
  ledstrip.show();  // Clear the strip for use
}

#if PRINT_FPS
uint16_t fpsCounter = 0;
uint32_t secondTimer = 0;
#endif

void loop()
{
  // Read data over socket
  int packetSize = port.parsePacket();
  // If packets have been received, interpret the command
  if (packetSize)
  {
    int len = port.read(packetBuffer, BUFFER_LEN);
    for (int i = 0; i < len; i += 4)
    {
      packetBuffer[len] = 0;
      N = packetBuffer[i];
      ledstrip.setPixelColor(N, (uint8_t)packetBuffer[i + 1], (uint8_t)packetBuffer[i + 2], (uint8_t)packetBuffer[i + 3]); // N is the pixel number
    }
    ledstrip.setBrightness(150); // Perhaps set another pin reading to control this live
    ledstrip.show();
#if PRINT_FPS
    fpsCounter++;
    Serial.print("/"); // Monitors connection(shows jumps/jitters in packets)
#endif
  }
#if PRINT_FPS
  if (millis() - secondTimer >= 1000U)
  {
    secondTimer = millis();
    Serial.print("FPS: ");
    Serial.println(fpsCounter);
    fpsCounter = 0;
  }
#endif
}
