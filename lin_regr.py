from os import walk
from sklearn import model_selection
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pickle

def training_function_across_seasons(fn,mod_lin_regr):

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

	print(len(X))
	print(len(y))
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
	plt.figure()
	plt.scatter(X_train, y_train, color='red')
	plt.scatter(X_test, y_test, color='green')
	plt.title('Plotted Data'+" "+fn)
	plt.xlabel('Games Played')
	plt.ylabel('Wins-Above-Average')
	# plt.show()

	plt.figure()
	plt.scatter(X_train, y_train, color='green')
	plt.plot(X_train, y_train_pred, color='black', linewidth=4)
	plt.title('Training data'+" "+fn)
	plt.xlabel('Games Played')
	plt.ylabel('Wins-Above-Average')
	# plt.show()

	y_test_pred = linear_regressor.predict(X_test)
	plt.figure()
	plt.scatter(X_test, y_test, color='green')
	plt.plot(X_test, y_test_pred, color='black', linewidth=4)
	plt.xlabel('Games Played')
	plt.ylabel('Wins-Above-Average')
	plt.title('Test data'+" "+fn)
	# plt.show()


	# Measure performance
	print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2)) 
	print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2)) 
	print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)) 
	print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2)) 
	print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))


	y_test_pred_new = mod_lin_regr.predict(X_test)
	print("New mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2)) 

	scores = model_selection.cross_val_score(estimator=linear_regressor,
							X=X_train,
							y=y_train,
							cv=6,
							n_jobs=1)

	print('CV accuracy scores: %s' % scores)
	print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

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

	print(len(X))
	print(len(y))
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
	plt.figure()
	plt.scatter(X_train, y_train, color='red')
	plt.scatter(X_test, y_test, color='green')
	plt.title('Plotted Data'+" "+fn)
	plt.xlabel('Games Played')
	plt.ylabel('Wins-Above-Average')
	# plt.show()

	plt.figure()
	plt.scatter(X_train, y_train, color='green')
	plt.plot(X_train, y_train_pred, color='black', linewidth=4)
	plt.title('Training data'+" "+fn)
	plt.xlabel('Games Played')
	plt.ylabel('Wins-Above-Average')
	# plt.show()

	y_test_pred = linear_regressor.predict(X_test)
	plt.figure()
	plt.scatter(X_test, y_test, color='green')
	plt.plot(X_test, y_test_pred, color='black', linewidth=4)
	plt.xlabel('Games Played')
	plt.ylabel('Wins-Above-Average')
	plt.title('Test data'+" "+fn)
	# plt.show()


	# Measure performance
	print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2)) 
	print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2)) 
	print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)) 
	print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2)) 
	print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))


	y_test_pred_new = linear_regressor.predict(X_test)
	print("New mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2)) 

	scores = model_selection.cross_val_score(estimator=linear_regressor,
							X=X_train,
							y=y_train,
							cv=6,
							n_jobs=1)

	print('CV accuracy scores: %s' % scores)
	print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

	return linear_regressor,scores


top_performer_files = []
path_to_top_performer_files = "/home/user/baseball_stats_folder"

for (dirpath, dirnames, filenames) in walk(path_to_top_performer_files):
	top_performer_files.extend(filenames)
	break
top_performer_files = sorted(top_performer_files)	
# print(top_performer_files)

mapping_player_model = {}

highest_CV = 0
model_linregr = "" 
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
	# print(i)	
	lin_reg,scores = training_function(i)
	mapping_player_model[i] = (lin_reg,np.mean(scores),np.std(scores))
	if np.mean(scores) > highest_CV:
		# Model persistence
		output_model_file = "3_model_linear_regr.pkl"

		with open(output_model_file, 'wb') as f:
			pickle.dump(lin_reg, f)

		with open(output_model_file, 'rb') as f:
			model_linregr = pickle.load(f)

# print(mapping_player_model)	

highest_CV = 0
save_index_of_best_model = ""
for i in top_performer_files:
	if "2013" in i:
		continue
	if "2019" in i:
		continue	
	lin_reg,scores = training_function_across_seasons(i,model_linregr)
	mapping_player_model[i] = (lin_reg,np.mean(scores),np.std(scores))
	if np.mean(scores) > highest_CV:
		# Model persistence
		output_model_file = "3_model_linear_regr.pkl"

		with open(output_model_file, 'wb') as f:
			pickle.dump(lin_reg, f)
		save_index_of_best_model = i	
		# with open(output_model_file, 'rb') as f:
		# 	model_linregr = pickle.load(f)

print(mapping_player_model[save_index_of_best_model])
import sys
lowest_std = sys.float_info.max 
mvp = ""
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
	output_model_file = "3_model_linear_regr.pkl"

	with open(output_model_file, 'wb') as f:
		pickle.dump(lin_reg, f)	
	with open(output_model_file, 'rb') as f:
		model_linregr = pickle.load(f)

	lin_reg,scores = training_function_across_seasons(i,model_linregr)
	if np.std(scores) < lowest_std:
		mvp = i
print(mvp)



# from sklearn.preprocessing import PolynomialFeatures

# X = []
# y = []
# with open(filename, 'r') as f:
#     for line in f.readlines():
#         xt, yt = [float(i) for i in line.split(',')]
#         X.append(xt)
#         y.append(yt)
        
# # Train/test split
# num_training = int(0.75 * len(X))
# num_test = len(X) - num_training

# import numpy as np

# # Training data
# X_train = np.array(X[:num_training]).reshape((num_training,1))
# y_train = np.array(y[:num_training])


# # Test data
# X_test = np.array(X[num_training:]).reshape((num_test,1))
# y_test = np.array(y[num_training:])

# lr = linear_model.LinearRegression()
# pr = linear_model.LinearRegression()
# quadratric = PolynomialFeatures(degree=2)
# X_quad = quadratric.fit_transform(X)

# lr.fit(X,y)
# X_fit = np.arange(0,170,10)[:, np.newaxis]
# y_lin_fit = lr.predict(X_fit)
# pr.fit(X_quad, y)
# y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))
# plt.scatter(X, y, label='training points')
# plt.plot(X_fit, y_lin_fit,
# 		label='linear fit', linestyle='--')
# plt.plot(X_fit, y_quad_fit,
# 		label='quadratic fit')
# plt.legend(loc='upper left')
# plt.show()
