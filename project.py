"""
@author: Cameron Nakakura
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# getting data from csv file
data = np.genfromtxt('fruitohms.csv', delimiter=",", \
                     skip_header=1, usecols=(1,2))
x = data[:,0] # apparent juice content (%)
y = data[:,1] # resistance (ohms)

# reshaping arrays
x = x.reshape(-1,1)
y = y.reshape(-1,1)

# instantiate new LinearRegression object
lr = LinearRegression()

# splitting data into training and testing data; fitting the data
x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=0.2)
lr.fit(x_train, y_train)

# generate prediction vals (line of best fit)
y_pred = lr.predict(x)

# calculating coefficient of determination
R2 = r2_score(y, y_pred)
print("R2 =", R2)

# calculating RMSE
RMSE = mean_squared_error(y, y_pred)**(1/2)
print("RMSE =", RMSE)

# plotting data
plt.scatter(x, y)
plt.title("Resistance vs. Juice Content")
plt.xlabel("Apparent juice content (%)")
plt.ylabel("Resistance (ohms)")
plt.show()

# plotting predictions
plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test, color='r')
plt.plot(x, y_pred, color='r')
plt.title("Resistance vs. Juice Content")
plt.xlabel("Apparent juice content (%)")
plt.ylabel("Resistance (ohms)")
plt.show()