import requests
import pandas as pd 
import datetime
import io
from bs4 import BeautifulSoup
from zipfile import ZipFile

def bwar_bat_interval(day):
	
	return_all = False
	# format war_archive-YYYY-MM-DD
	file = "war_archive-"+day
	zip_file = file+".zip"
	url = "http://www.baseball-reference.com/data/"+zip_file
	print("Retrieving file...\n")
	s = requests.get(url)

	with open(zip_file, 'wb') as code:
		code.write(s.content)

	with ZipFile(zip_file) as myzip:
	    with myzip.open('war_daily_bat.txt') as myfile:
	    	c = pd.read_csv(io.StringIO(myfile.read().decode('utf-8')))

	# print("Retrieved file...\nUnzipping...\n")
	# unzipped_file = zipfile.ZipFile(s.decode('utf-8'), 'r')
	# unzipped_file.extractall("/home/user")
	# unzipped_file.close()
	# s = open("/home/user"+file)
	# c = pc.read_csv(io.StringIO(s))

	cols_to_keep = ['name_common', 'mlb_ID', 'player_ID', 'year_ID', 'team_ID', 'stint_ID', 'lg_ID', 
					'pitcher','G', 'PA', 'salary', 'runs_above_avg', 'runs_above_avg_off','runs_above_avg_def',
					'WAR_rep','WAA','WAR']
	return c[cols_to_keep]				

print("Finding data")
lf = bwar_bat_interval("2013-12-31")
print(lf)	
