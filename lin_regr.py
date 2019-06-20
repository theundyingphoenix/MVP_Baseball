filename = "baseball_stats_folder/2013_goldspa01"

X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)

from random import shuffle
d = {}
for i in range(len(X)):
	d[i] = y[i]

shuffle(X)

y = []
for i in X:
	y.append(d[i])
	        
# Train/test split
num_training = int(0.75 * len(X))
num_test = len(X) - num_training

import numpy as np

# Training data
X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])


# Test data
X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])


# Create linear regression object
from sklearn import linear_model

linear_regressor = linear_model.LinearRegression()

# Train the model using the training sets
linear_regressor.fit(X_train, y_train)

# Predict the output
y_train_pred = linear_regressor.predict(X_train)

# Plot outputs
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(X_train, y_train, color='red')
plt.scatter(X_test, y_test, color='green')
plt.title('Plotted Data')
plt.show()

plt.figure()
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')
plt.show()

y_test_pred = linear_regressor.predict(X_test)
plt.figure()
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.title('Test data')
plt.show()


# Measure performance
import sklearn.metrics as sm

print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2)) 
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Model persistence
import pickle

output_model_file = "3_model_linear_regr.pkl"

with open(output_model_file, 'wb') as f:
    pickle.dump(linear_regressor, f)

with open(output_model_file, 'rb') as f:
    model_linregr = pickle.load(f)

y_test_pred_new = model_linregr.predict(X_test)
print("New mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2)) 


from sklearn import model_selection
scores = model_selection.cross_val_score(estimator=linear_regressor,
						X=X_train,
						y=y_train,
						cv=6,
						n_jobs=1)

print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


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
