from gpiozero import LED
from time import sleep

# Initialize the LED on GPIO pin 17 (BCM numbering)
led = LED(18)

print("Blinking LED... Press Ctrl+C to stop.")

try:
    while True:
        led.on()       	# Turn the LED on
        print("LED On")
        sleep(1)       	# Wait for 1 second
        led.off()      	# Turn the LED off
        print("LED Off")
        sleep(1)       	# Wait for 1 second

except KeyboardInterrupt:
    # Smoothly handle exiting the program when Ctrl+C is pressed
    print("\nProgram stopped.")
