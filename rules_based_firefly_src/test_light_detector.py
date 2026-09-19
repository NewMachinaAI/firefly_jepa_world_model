from gpiozero import LineSensor
from signal import pause
import time

# Initialize GPIO 17 as a digital input
# Note: LM393 DO outputs LOW when light is detected, HIGH when dark
sensor = LineSensor(17)

def light_detected():
    print(f"[{time.strftime('%H:%M:%S')}] Light Detected (Environment Bright)")

def dark_detected():
    print(f"[{time.strftime('%H:%M:%S')}] Darkness Detected (Environment Dark)")

# Attach callbacks to sensor state changes
sensor.when_line = dark_detected      # High signal = Dark
sensor.when_no_line = light_detected  # Low signal = Light

print("LDR Light Sensor Active. Cover/uncover sensor to test...")
print("Press Ctrl+C to exit.")
print("Initialize sensor state..." + str(sensor.value))

try:
    pause()
except KeyboardInterrupt:
    print("\nProgram stopped.")