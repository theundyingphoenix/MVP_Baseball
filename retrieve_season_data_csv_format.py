import pymongo
import pandas as pd 
import numpy as np 
import sys, re, os

# create connection to database
client = pymongo.MongoClient('localhost', 27017) # change 'localhost' to MongoDB holding stats

# season interested in
database_names = client.database_names()
year = '2019'
year_database = ""
all_years = []
for db in database_names:
	if year in db:
		year_database = db
	if 'baseball' in db:
		all_years.append(db)	

# number of players that had WAR > 6.5 by end of season 
# (i.e. top-performers) pymongo code 
year_database = client[year_database]
archives = sorted(year_database.list_collection_names())


end_of_season_col = archives[-1]
end_of_season_col = year_database[end_of_season_col]

number_of_top_performers = 0
top_performer_IDs = []
for player in end_of_season_col.find({"WAR": {"$gte": 3.0}}):
	top_performer_IDs.append(player['player_ID'])
	number_of_top_performers += 1

# file to save the stats of top-performing players
path = os.getcwd()
path = path+"/baseball_stats_folder" 
for i in top_performer_IDs:
	filename = year+"_"+i

	f = open(path+"/"+filename, 'w')
	# going through each archive of the season and pulling the (G,WAR)-pair
	# write it to the file object (sequentially)
	for a in archives:
		# in the collection
		c = year_database[a]
		p = c.find_one({"player_ID": i})
		if p is None:
			continue
		g = p['G']
		war = p['WAR']
		pair = str(g)+","+str(war)+"\n"
		f.write(pair)
	f.close()	




