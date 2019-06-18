import os
import requests
import pandas as pd 
import datetime
import io
from bs4 import BeautifulSoup
from zipfile import ZipFile
from pymongo import MongoClient
import json

def bwar_bat_interval(day):
	
	return_all = False
	# format war_archive-YYYY-MM-DD
	file = "war_archive-"+day
	zip_file = file+".zip"
	url = "http://www.baseball-reference.com/data/"+zip_file
	print("Retrieving data from 2013-03-29 to "+zip_file+"...\n")
	
	s = requests.get(url)
	if s.status_code == 404:
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

client = MongoClient('localhost', 27017)
print("Acquired mongodb client...\n")

# years = ['2013','2014','2015','2016','2017','2018']

# collection = 'war_archive-2017-05-08'
# db_cm = db[collection]


# import the dates to have the war_daily_bat.txt
under_interval = []
interval = []
print("Opening seasons.txt & season.txt...\n")
f = open("/home/user/seasons.txt", 'r')
ff = open("/home/user/season.txt", 'r')

# remove the \n
l = f.readline()[:-1]
ll = ff.readline()[:-1]
while l:
	interval.append(l)
	under_interval.append(ll)
	l = f.readline()[:-1]
	ll = ff.readline()[:-1]
f.close()
ff.close()	

print("Collected the dates of stats to import...\n")

for i in range(len(interval)):
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
	print(lf.head(5))

	json_data = json.loads(lf.to_json(orient='records'))
	print(json_data)
	db_cm.remove()

	
	db_cm.insert(json_data)	
	print("Added to MongoDB...")
	os.remove("war_archive-"+interval[i]+".zip")
