import os
import requests
import pandas as pd 
import datetime
import io
from bs4 import BeautifulSoup
from zipfile import ZipFile
import json
import pymongo

def catalogue_database(client):
	client = pymongo.MongoClient('localhost', 27017)
	list_of_databases = client.database_names()


	db_prefix = 'baseball_stats_'
	collection_prefix = 'war_archive_'

	dates = []
	underscored_dates = []
	f = open("/home/user/seasons.txt", 'r')
	line = f.readline()
	while line:
		dates.append(line)
		line = f.readline()

	years_present = {}
	for u in range(len(dates)):
		year = dates[u][:4]	
		db = db_prefix+year
	
		for i in range(len(list_of_databases)):
			if year in list_of_databases[i]:
				years_present[year] = True

	# for the year's db we want to mark the values in underscored_dates
	# as True to catalogue the archives present, then take the latest
	# date as the place to start

	for k in years_present:
		# access the db
		db = db_prefix+k
		db = pymongo.database.Database(client,db)
		archived_dates = db.list_collection_names()
		for a in archived_dates:
			underscored_dates.append(a)
		
	return sorted(underscored_dates)		



def bwar_bat_interval(day):
	
	return_all = False
	# format war_archive-YYYY-MM-DD
	file = "war_archive-"+day
	zip_file = file+".zip"
	url = "http://www.baseball-reference.com/data/"+zip_file
	print("Retrieving data from "+zip_file+"...\n")
	
	s = requests.get(url)

	if s.status_code == 200:
		print(s.status_code)
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

client = pymongo.MongoClient('localhost', 27017)
print("Acquired mongodb client...\n")

# import the dates to have the war_daily_bat.txt
# for storing archives in correct format
under_interval = [] 
# for the list of dates to traverse in url format
interval = [] 
print("Opening seasons.txt & season.txt...\n")
f = open("/home/user/seasons.txt", 'r')
ff = open("/home/user/season.txt", 'r')


# remove the \n
l = f.readline()[:-1]
while l:
	interval.append(l)
	under_interval.append(l.replace("-","_"))
	l = f.readline()[:-1]
f.close()

print("Collected the dates of stats to import...\n")

for i in range(len(interval)):
	# to prevent process from starting from the beginning
	present_archives = catalogue_database(client)
	most_recently_added = present_archives[-1]

	db = client['baseball_stats_'+interval[i][:4]]
	collection = 'war_archive_'+under_interval[i]
	db_cm = db[collection]


	print("Finding data for "+interval[i])
	lf = bwar_bat_interval(interval[i])	

	lf = lf[lf.year_ID >= 2013]
	lf = lf[lf.lg_ID != "AL"]

	if lf.empty:
		continue		
	
	print("DataFrame collected")
	
	json_data = json.loads(lf.to_json(orient='records'))
	db_cm.remove()
	db_cm.insert(json_data)	
	print("JSON data added to MongoDB...")
	saved_checkpoint = open("/home/user/checkpoint.txt", 'w')
	saved_checkpoint.write(interval[i])
	saved_checkpoint.close()
	os.remove("war_archive-"+interval[i]+".zip")
