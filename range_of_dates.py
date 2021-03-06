import datetime

''' Provide a list of dates to search the 
    baseball-reference site in appropriate format.'''

list_of_dates = []

date1 = '2013-03-01'
date2 = '2018-10-31'

months_disregard = ['01','02','11','12']

start = datetime.datetime.strptime(date1, '%Y-%m-%d')
end = datetime.datetime.strptime(date2, '%Y-%m-%d')

step = datetime.timedelta(days=1)
while start <= end:
	do_not_add_date = False
	
	for i in months_disregard:
		if i == start.strftime("%m"):
			do_not_add_date = True
	
	if not do_not_add_date:
		list_of_dates.append(str(start.date()))
	
	start += step

for idx, item in enumerate(list_of_dates):
	print(item)
	
	
	
