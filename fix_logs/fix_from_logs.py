#!/usr/bin/env python

import re
import sys
import time
import argparse
import datetime
import pandas as pd

from telemetry import Telemetry

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, nargs='+', help="Log files to process")
parser.add_argument('-t', '--time', type=float, help="Hours to adjunt time zone, ex: -3, +4, etc.")
args = parser.parse_args()

re_sample = re.compile(r"((?:0x....,){2}(?:0x0043,){3}0x000.,(?:0x....,){6,8})(?=(?:0x....,){2}(?:0x0043,){3}0x000.)")

# The file name is the first and only parameter
fname = args.file[0]
#fname = "../data/20180614-DIA/SUCHAI_20180614_130248.txt"

# The Object that parses the telemetry
tm = Telemetry(date=time.asctime())
tm.set_payload(str(3))

# Read the file and
lines = []
with open(fname) as datafile:
    for line in datafile:
        # Delete the debug information, type and number like: "[2018-09-10 13:55:08.296842][tm] 0x0300,0xFFA0,"
        lines.append(line[47:])

# Apply the regexp to find samples like
# "0x24E3,0x4A54,0x0043,0x0043,0x0043,0x0005,0x0003,0x0036,0x0002,0x0047,0x0002,0x004F,0x0000,0x000A,"
# The regexp is evaluated over plain string, replacing \n by ","
# The regexp itself filter incomplete frames ;)
lines = "".join(lines)
lines = lines.replace("\n", ",")
samples = re_sample.findall(lines)

# Let filter some incomplete samples from broken frames
samples = list(filter(lambda l: not ("0x0043,0x0043,0x0043,0x0005" in l and len(l) < 98), samples))

#print(samples)

# Now add the samples to the telemetry parses to create de DataFrame
for i, line in enumerate(samples):
    if line[-1] == ",":
        line = line[:-1]
    data = line.split(",")
    tm.set_data(data, hex(i))

# Finally save the file
df = tm.to_dataframe()
if args.time:
    t = pd.to_datetime(df["time"], errors='coerce')
    dt = datetime.timedelta(hours=float(args.time))
    df["time"] = t+dt
df.to_csv(fname+".csv")

## TAIL EXAMPLE
"""
0x53B5,0x4A74,0x0043,0x0043,0x0043,0x0001,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,
0x0000,0x0000,0x0043,0x0043,0x0043,0x0002,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,
0x0000,0x0000,0x0043,0x0043,0x0043,0x0003,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,
0x0000,0x0000,0x0043,0x0043,0x0043,0x0004,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0xFFFE,

or


"""
