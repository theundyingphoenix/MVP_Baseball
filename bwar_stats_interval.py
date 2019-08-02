import os
import requests
import pandas as pd 
import datetime
import io
from bs4 import BeautifulSoup
from zipfile import ZipFile
import json
import pymongo

'''  '''

# Method to check which archives are already present
def catalogue_database(client):
	# Create connection to MongoDB
	client = pymongo.MongoClient('localhost', 27017)
	# Grab the list of databases present
	list_of_databases = client.database_names()

	# Prefixes in correct format
	db_prefix = 'baseball_stats_'
	collection_prefix = 'war_archive_'

	dates = []
	underscored_dates = []
	f = open(os.getcwd()+"/season.txt", 'r')
	line = f.readline()
	while line:
		dates.append(line)
		line = f.readline()

	years_present = {}
	for i in range(len(dates)):
		year = dates[i][:4]	
		db = db_prefix+year
	
		for j in range(len(list_of_databases)):
			if year in list_of_databases[j]:
				years_present[year] = True

	# for the year's db we want to mark the values in underscored_dates
	# as True to catalogue the archives present, then take the latest
	# date as the place to start

	# Archive all the dates that are present in the database
	for k in years_present:
		# access the db
		db = db_prefix+k
		# Grab a specific database
		db = pymongo.database.Database(client,db)
		archived_dates = db.list_collection_names()
		for a in archived_dates:
			underscored_dates.append(a)
		
	return sorted(underscored_dates)		



def bwar_bat_interval(day):
	
	# format war_archive-YYYY-MM-DD
	file = "war_archive-"+day
	zip_file = file+".zip"
	url = "http://www.baseball-reference.com/data/"+zip_file
	print("Retrieving data from "+zip_file+"...\n")
	
	s = requests.get(url)

	if s.status_code == 200:
		print(s.status_code)
		print("File exists.")
	else:
		return None	

	with open(zip_file, 'wb') as code:
		code.write(s.content)

	with ZipFile(zip_file) as myzip:
	    with myzip.open('war_daily_bat.txt') as myfile:
	    	c = pd.read_csv(io.StringIO(myfile.read().decode('utf-8')))
	    	

	cols_to_keep = ['name_common', 'player_ID', 'year_ID', 'team_ID', 'stint_ID', 'lg_ID', 
					'pitcher','G', 'PA', 'salary', 'runs_above_avg', 'runs_above_avg_off','runs_above_avg_def',
					'WAR_rep','WAA','WAR']
	
	return c[cols_to_keep]				


# Create a connection with the database
client = pymongo.MongoClient('localhost', 27017)
print("Acquired mongodb client...\n")

# for the list of dates to traverse in url format
interval = [] 

print("Opening season.txt...\n")
f = open(os.getcwd()+"/season.txt", 'r')
# remove the \n
l = f.readline()[:-1]
while l:
	interval.append(l)
	l = f.readline()[:-1]
f.close()

print("Collected the dates of stats to import...\n")

catalogued = False
update = ""
update_index = -1
no_db_filled = False
for i in range(len(interval)):
	# to prevent process from starting from the beginning
	# in the even program shuts down
	if not catalogued:
		present_archives = catalogue_database(client)
		if present_archives:
			most_recently_added = present_archives[-1]
			update = most_recently_added
			catalogued = True
			continue

	if not update in interval[i]:
		continue	

	year = interval[i][:4]
	db = client['baseball_stats_'+year]
	collection = 'war_archive_'+under_interval[i]
	db_cm = db[collection]

	print("Finding data for "+interval[i])
	lf = bwar_bat_interval(interval[i])	
	if lf is None:
		continue
	
	lf = lf[lf.year_ID >= int(year)]
	lf = lf[lf.lg_ID != "AL"]
	if lf.empty:
		continue

	print("DataFrame collected")
	
	json_data = json.loads(lf.to_json(orient='records'))
	db_cm.remove()
	db_cm.insert(json_data)	
	print("JSON data added to MongoDB...")
	os.remove("war_archive-"+interval[i]+".zip")
