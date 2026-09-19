import random
import time
from gpiozero import DigitalInputDevice, LED

# Initialize GPIO pins
# adjust active_state if your module reads HIGH during daylight
sensor = DigitalInputDevice(17, pull_up=False)
firefly_led = LED(18)

print("Firefly simulation running. Press Ctrl+C to exit.")

try:
    while True:
        # Check if sensor detects darkness
        # Default LM393 modules output HIGH (1) in dark, LOW (0) in light
        is_dark = sensor.value == 1

        if is_dark:
            print("Darkness detected: Flashing firefly light...")
            firefly_led.on()
            time.sleep(1.0)  # On for 1 second
            firefly_led.off()

            # Random pause between 2 and 5 seconds before checking again
            delay = random.uniform(2.0, 5.0)
            time.sleep(delay)
        else:
            print("Daylight detected: Firefly sleeping...")
            firefly_led.off()
            time.sleep(1.0)  # Check sensor every second during daylight

except KeyboardInterrupt:
    print("\nExiting script and cleaning up GPIO.")
    firefly_led.off()