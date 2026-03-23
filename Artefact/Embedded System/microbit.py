from microbit import *

while True:
    # Read the internal temperature
    temp = temperature()
    # Send it over USB serial
    print(temp)
    # Wait 5 seconds (5000 milliseconds)
    sleep(5000)