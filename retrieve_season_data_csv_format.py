import pymongo
import pandas as pd 
import numpy as np 
import sys, re, os

# create connection to database
client = pymongo.MongoClient('10.109.29.134', 27017)

# season interested in
database_names = client.database_names()
year = '2013'
year_database = database_names.find('2013')
print(year_database)

# number of players that had WAR > 6.5 by end of season 
# (i.e. top-performers) pymongo code 

# number_of_top_performers = 



# file to save the stats of top-performing players
# f = open('/home/user/2013_GvWAR.txt', 'w')


# from each top-performer we need the list of (G,WAR)-pairs in each
# archive for the whole season -- A way to store separate files for 
# each regressor model, then compare those models
# !!! Need a way to associate a file to a player and regressor !!!



# create regressor and follow code in the Packt Publ. Books in
# another python program or this one??


