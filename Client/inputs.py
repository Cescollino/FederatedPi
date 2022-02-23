
from smbus2 import SMBus
import time

# RPi Channel 1
channel = 1

bus = SMBus(channel)
# ADS1115 address and registers
address = 0x48
reg_config = 0x01
 
# Config value:
# - Single conversion
# - A0 input
# - 4.096V reference
config = [0xC2, 0xB3]

while True:
  reg_conversion = 0x00

  # Start conversion
  bus.write_i2c_block_data(address, reg_config, config)
  # Wait for conversion
  time.sleep(0.01)
  # Read 16-bit result
  result = bus.read_i2c_block_data(address, reg_conversion, 2)
  # Convert from 2-complement
  value = ((result[0] & 0xFF) << 8) | (result[1] & 0xFF)
  if value & 0x8000 != 0:
    value -= 1 << 16
  # Convert value to voltage
  v = value * 4.096 / 32768
  print("A0:", v)
  # Wait a second to start again
  time.sleep(1)

  # ADS1115 address and registers
  address = 0x48
  reg_config = 0x01
  reg_conversion = 0x01

  # Start conversion
  bus.write_i2c_block_data(address, reg_config, config)
  # Wait for conversion
  time.sleep(0.01)
  # Read 16-bit result
  result = bus.read_i2c_block_data(address, reg_conversion, 2)
  # Convert from 2-complement
  value = ((result[0] & 0xFF) << 8) | (result[1] & 0xFF)
  if value & 0x8000 != 0:
    value -= 1 << 16
  # Convert value to voltage
  v = value * 4.096 / 32768
  print("A1:", v)
  # Wait a second to start again
  time.sleep(1)

