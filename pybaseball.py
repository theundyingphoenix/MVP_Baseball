import pandas as pd 
import scipy
import numpy as np 
import pybaseball
from pybaseball import statcast
from pybaseball import bwar_bat

print("Collecting statcast data...")

data = statcast(start_dt='2017-06-24', end_dt='2017-06-27')
print(data.head(2))

print("Starting bWAR collection...")

data = bwar_bat()
print(data.head(50))

print("\n")

print("Creating DataFrame...")
df = pd.DataFrame(data)
print("\n****DATAFRAME****\n")
print(df)
print("\n")


stopping_year = 2013
df = df[df.year_ID >= 2013]
df = df[df.lg_ID != "AL"]
df = df[df.pitcher == "N"]
print("***DATAFRAME***\n\n")
print(df)
print("EOP")