from os import walk
from sklearn import model_selection
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pickle

def training_function_across_seasons(fn,mod_lin_regr,num):
	
	filename = "baseball_stats_folder/"+fn

	X = []
	y = []


	with open(filename, 'r') as f:
		for line in f.readlines():
			# some values will have None
			t = line.split(',')
			if 'None' in t[0] or 'None' in t[1]:
				continue
			xt = float(t[0])
			yt = float(t[1])
			X.append(xt)
			y.append(yt)

	from random import shuffle
	d = {}
	for i in X:
		index = X.index(i)
		d[i] = y[index]

	shuffle(X)

	y = []
	for i in X:
		y.append(d[i])

	# Train/test split
	num_training = int(0.75 * len(X))
	num_test = len(X) - num_training



	# Training data
	X_train = np.array(X[:num_training]).reshape((num_training,1))
	y_train = np.array(y[:num_training])


	# Test data
	X_test = np.array(X[num_training:]).reshape((num_test,1))
	y_test = np.array(y[num_training:])


	# Create linear regression object
	linear_regressor = mod_lin_regr

	# Train the model using the training sets
	linear_regressor.fit(X_train, y_train)

	# Predict the output
	y_train_pred = linear_regressor.predict(X_train)

	# Plot outputs
	if num == 1:
		plt.figure()
		plt.scatter(X_train, y_train, color='red')
		plt.scatter(X_test, y_test, color='green')
		plt.title('Plotted Data'+" "+fn)
		plt.xlabel('Games Played')
		plt.ylabel('Wins-Above-Average')
		plt.show()

		plt.figure()
		plt.scatter(X_train, y_train, color='green')
		plt.plot(X_train, y_train_pred, color='black', linewidth=4)
		plt.title('Training data'+" "+fn)
		plt.xlabel('Games Played')
		plt.ylabel('Wins-Above-Average')
		plt.show()

	y_test_pred = linear_regressor.predict(X_test)
	
	if num == 1:
		plt.figure()
		plt.scatter(X_test, y_test, color='green')
		plt.plot(X_test, y_test_pred, color='black', linewidth=4)
		plt.xlabel('Games Played')
		plt.ylabel('Wins-Above-Average')
		plt.title('Test data'+" "+fn)
		plt.show()


	# # Measure performance
	# print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2)) 
	# print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2)) 
	# print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)) 
	# print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2)) 
	# print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))


	y_test_pred_new = mod_lin_regr.predict(X_test)
	# print("New mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2)) 

	scores = model_selection.cross_val_score(estimator=linear_regressor,
							X=X_train,
							y=y_train,
							cv=10,
							n_jobs=1)

	# print('CV accuracy scores: %s' % scores)
	# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

	return linear_regressor,scores

def training_function(fn):

	filename = "baseball_stats_folder/"+fn

	X = []
	y = []


	with open(filename, 'r') as f:
		for line in f.readlines():
			# some values will have None
			t = line.split(',')
			if 'None' in t[0] or 'None' in t[1]:
				continue
			xt = float(t[0])
			yt = float(t[1])
			X.append(xt)
			y.append(yt)

	from random import shuffle
	d = {}
	for i in X:
		index = X.index(i)
		d[i] = y[index]

	shuffle(X)

	y = []
	for i in X:
		y.append(d[i])

	# Train/test split
	num_training = int(0.75 * len(X))
	num_test = len(X) - num_training



	# Training data
	X_train = np.array(X[:num_training]).reshape((num_training,1))
	y_train = np.array(y[:num_training])


	# Test data
	X_test = np.array(X[num_training:]).reshape((num_test,1))
	y_test = np.array(y[num_training:])


	# Create linear regression object
	linear_regressor = linear_model.LinearRegression()

	# Train the model using the training sets
	linear_regressor.fit(X_train, y_train)

	# Predict the output
	y_train_pred = linear_regressor.predict(X_train)

	# Plot outputs
	# plt.figure()
	# plt.scatter(X_train, y_train, color='red')
	# plt.scatter(X_test, y_test, color='green')
	# plt.title('Plotted Data'+" "+fn)
	# plt.xlabel('Games Played')
	# plt.ylabel('Wins-Above-Average')
	# # plt.show()

	# plt.figure()
	# plt.scatter(X_train, y_train, color='green')
	# plt.plot(X_train, y_train_pred, color='black', linewidth=4)
	# plt.title('Training data'+" "+fn)
	# plt.xlabel('Games Played')
	# plt.ylabel('Wins-Above-Average')
	# # plt.show()

	y_test_pred = linear_regressor.predict(X_test)
	# plt.figure()
	# plt.scatter(X_test, y_test, color='green')
	# plt.plot(X_test, y_test_pred, color='black', linewidth=4)
	# plt.xlabel('Games Played')
	# plt.ylabel('Wins-Above-Average')
	# plt.title('Test data'+" "+fn)
	# # plt.show()


	# Measure performance
	# print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2)) 
	# print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2)) 
	# print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)) 
	# print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2)) 
	# print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))


	y_test_pred_new = linear_regressor.predict(X_test)
	# print("New mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2)) 

	scores = model_selection.cross_val_score(estimator=linear_regressor,
							X=X_train,
							y=y_train,
							cv=10,
							n_jobs=1)

	# print('CV accuracy scores: %s' % scores)
	# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

	return linear_regressor,scores


top_performer_files = []
path_to_top_performer_files = "/home/user/baseball_stats_folder"

for (dirpath, dirnames, filenames) in walk(path_to_top_performer_files):
	top_performer_files.extend(filenames)
	break
top_performer_files = sorted(top_performer_files)	


# train the 2013 starting model with CV values
model_linregr = "" 
best_init_model = {}
for i in top_performer_files:
	if "2014" in i:
		continue
	if "2015" in i:
		continue
	if "2016" in i:
		continue
	if "2017" in i:
		continue
	if "2018" in i:
		continue			
	if "2019" in i:
		continue
	if i is None:
		continue

	lin_reg,scores = training_function(i)
	best_init_model[i] = (lin_reg,scores)
	

for i in best_init_model.keys():
	print(i+" Std: "+str(np.std(best_init_model[i][1]))+" Mean: "+str(np.mean(best_init_model[i][1])))

key_max = max(best_init_model.keys(), key=(lambda k: np.mean(best_init_model[k][1])))
print("\n\nBest init model: "+key_max)
output_model_file = "3_model_linear_regr.pkl"

with open(output_model_file, 'wb') as f:
	pickle.dump(best_init_model[key_max][0], f)

# training with other seasons  to make a better model
highest_CV = 0
save_index_of_best_model = ""
best_season_trained_model = {}
for i in top_performer_files:
	if "2013" in i:
		continue
	if "2019" in i:
		continue		

	# revising model
	output_model_file = "3_model_linear_regr.pkl"

	with open(output_model_file, 'rb') as f:
		model_linregr = pickle.load(f)
		
	lin_reg,scores = training_function_across_seasons(i,model_linregr,0)
	best_season_trained_model[i] = (lin_reg, scores)
	
print("\n\n")
# for i in best_season_trained_model.keys():
# 	print(i+" Std: "+str(np.std(best_season_trained_model[i][1]))+" Mean: "+str(np.mean(best_season_trained_model[i][1])))

key_max = max(best_season_trained_model.keys(), key=(lambda k: np.mean(best_season_trained_model[k][1])))
print("\n\nBest season model: "+key_max)
output_model_file = "best_model_linear_regr.pkl"

with open(output_model_file, 'wb') as f:
	pickle.dump(best_season_trained_model[key_max][0], f)




import sys
lowest_std = sys.float_info.max
mvp = ""

top_performer_files = []
path_to_top_performer_files = "/home/user/baseball_stats_folder/"

for (dirpath, dirnames, filenames) in walk(path_to_top_performer_files):
	top_performer_files.extend(filenames)
	break
top_performer_files = sorted(top_performer_files)	


mvp_dict_finder = {}
gp_war_dict = {}
# final testing set against a refined model
for i in top_performer_files:
	if "2014" in i:
		continue
	if "2015" in i:
		continue
	if "2016" in i:
		continue
	if "2017" in i:
		continue
	if "2018" in i:
		continue			
	if "2013" in i:
		continue
	if i is None:
		continue

	# Model persistence
	output_model_file = "best_model_linear_regr.pkl"

	with open(output_model_file, 'rb') as f:
		model_linregr = pickle.load(f)

	file = i	
	f_o = "/home/user/baseball_stats_folder/"+file
	f_o = open(f_o, 'r')
	line = f_o.readlines()[-1]
	total_games, WAR = line.split(',')
	lin_reg,scores = training_function_across_seasons(file,model_linregr,0)
	
	cont_check = float(np.std(scores))
	if cont_check <= 0.0:
		continue
	gp_war_dict[i] = line	
	mvp_dict_finder[i] = float(np.std(scores))
	

key_min = min(mvp_dict_finder.keys(), key=(lambda k: mvp_dict_finder[k]))
mvp = key_min
print(mvp+" Deviation: "+str(mvp_dict_finder[mvp])+"\n\n\n\n")
for i in mvp_dict_finder:
	print("ID: "+i+" Std: "+str(mvp_dict_finder[i]))
