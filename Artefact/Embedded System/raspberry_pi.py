import digitalio
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
import os
import time
import busio
import board
import serial

VAL_DRY = 54000         
VAL_WET = 27500         
SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 115200
TEMP_OFFSET = -2.0      
csv_path = "/home/channers/garden_data/plant_data.csv"

os.makedirs(os.path.dirname(csv_path), exist_ok=True)

spi = busio.SPI(clock=board.SCK, MOSI=board.MOSI, MISO=board.MISO)
cs = digitalio.DigitalInOut(board.D5)
mcp = MCP.MCP3008(spi, cs)
chan = AnalogIn(mcp, MCP.P0)

# Create a file in case it couldn't be found
if not os.path.exists(csv_path):
    with open(csv_path, "w") as f:
        f.write("Date,Time,Temp_C,Raw_Soil,Moisture_Percent\n")

def get_moisture_percent(raw):
    percentage = ((raw - VAL_DRY) / (VAL_WET - VAL_DRY)) * 100
    return max(0, min(100, percentage))

def get_microbit_temp():
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=7) as ser:
            line = ser.readline().decode('utf-8').strip()
            if line:
                clean_line = "".join(c for c in line if c in "0123456789.-")
                if clean_line:
                    return float(clean_line) + TEMP_OFFSET
    except Exception as e:
        print(f"Serial Error: {e}")
    return None

print(f"Plant monitering active, logging to {csv_path}")

while True:
    try:
        temp = get_microbit_temp()
        raw_soil_val = chan.value  
        moisture_pc = get_moisture_percent(raw_soil_val)

        if temp is not None:
            date_str = time.strftime('%Y-%m-%d')
            time_str = time.strftime('%H:%M:%S')

            with open(csv_path, "a") as f:
                # Date, Time, Temp, Raw, Percent
                f.write(f"{date_str},{time_str},{temp:.2f},{raw_soil_val},{moisture_pc:.1f}\n")
                f.flush()
                os.fsync(f.fileno())

            print(f"[{date_str} {time_str}] Temp: {temp:.1f}C | Raw: {raw_soil_val} | Moisture: {moisture_pc:.1f}%")
        else:
            print("Microbit not responding")

        time.sleep(3600)

    except KeyboardInterrupt:
        print("\nStopping")       
        break
    except Exception as e:
        print(f"Error: {e}")        # Print error
        time.sleep(60)