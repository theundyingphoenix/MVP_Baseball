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

	with open(zip_file, 'wb') as code:
		code.write(s.content)

	with ZipFile(zip_file) as myzip:
	    with myzip.open('war_daily_bat.txt') as myfile:
	    	c = pd.read_csv(io.StringIO(myfile.read().decode('utf-8')))


	cols_to_keep = ['name_common', 'mlb_ID', 'player_ID', 'year_ID', 'team_ID', 'stint_ID', 'lg_ID', 
					'pitcher','G', 'PA', 'salary', 'runs_above_avg', 'runs_above_avg_off','runs_above_avg_def',
					'WAR_rep','WAA','WAR']

	stopping_year = 2013
	c = c[c.year_ID >= 2013]
	c = c[c.lg_ID != "AL"]
	c = c[c.pitcher == "N"]	
	# c = c[c.name_common == "Lane Adams"]			
	return c[cols_to_keep]				

client = MongoClient('localhost', 27017)
db = client['baseball_stats']

years = ['2013','2014','2015','2016','2017','2018']


collection = 'war_stats_2017_05_08'
db_cm = db[collection]


# import the dates to have the war_daily_bat.txt
interval = []
f = open("/home/user/seasons", 'r')
l = f.readline()[:-1]
while l:
	interval.append(l)
	l = f.readline()[:-1]


print("Finding data for "+interval)
lf = bwar_bat_interval(interval)
print("DataFrame collected")

json_data = json.loads(lf.to_json(orient='records'))
db_cm.remove()
db_cm.insert(json_data)	

