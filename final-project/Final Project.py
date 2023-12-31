# Matthew, Liam, Daniel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
**********CREATE THE MODEL**********
'''

data = pd.read_csv("a6-predictive-models-lrgorman1/final-project/data.csv")
x = data[["Club Market Value"]].values
y = data ["Placement"].values

# Create your training and testing datasets:
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)
# Use reshape to turn the x values into 2D arrays:
xtrain = xtrain.reshape(-1, 1)
print(f"x {x}")
print(f"y {y}")
print(f"train {xtrain}")
print(f"train {xtest}")
print(f"train {ytrain}")
print(f"train {ytest}")
# Create the model
model = LinearRegression().fit(xtrain, ytrain)

# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(xtrain, ytrain)


# Print out the linear equation and r squared value:
print(f"Model's Linear Equation: y={coef[0]}x1 + {coef[1]}x2 + {intercept}")
print("R Squared value:", r_squared)
'''
**********TEST THE MODEL**********
'''
# reshape the xtest data into a 2D array
xtest = xtest.reshape(-1,1)
# get the predicted y values for the xtest values - returns an array of the results
predict = model.predict(xtest)
predict = np.around(predict, 2)
# round the value in the np array to 2 decimal places
print("\nTesting Linear Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index] # gets the actual y value from the ytest dataset
    predicted_y = predict[index] # gets the predicted y value from the predict variable
    x_coord = xtest[index] # gets the x value from the xtest dataset
    print("x value:", float(x_coord), "Predicted y value:", predicted_y, "Actual y value:", actual)
    print(f"PLacement: {x_coord[2]} Club Market Value: {x_coord[1]}  Actual: {actual} Predicted: {predicted_y}")
# Test the model by looping through all of the values in the xtest dataset
print("\nTesting Linear Model with Testing Data:")


'''
**********CREATE A VISUAL OF THE RESULTS**********
'''
plt.figure(figsize=(5,4))

plt.scatter(xtrain,ytrain, c="purple", label="Training Data")
plt.scatter(xtest, ytest, c="blue", label="Testing Data")

plt.scatter(xtest, predict, c="red", label="Predictions")

plt.xlabel("Club's value")
plt.ylabel("PLacement")
plt.title("PLacement by Club's value")
plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

plt.legend()
plt.show()

