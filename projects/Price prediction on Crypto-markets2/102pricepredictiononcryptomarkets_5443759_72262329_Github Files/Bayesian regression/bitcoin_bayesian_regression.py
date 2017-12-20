import statsmodels.formula.api as smf
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import math
import sys
import matplotlib
from matplotlib import pyplot as plt

# Set data path
data_path = "/Users/pablocorrea/Downloads/Bayesian-Regression-to-Predict-Bitcoin-Price-Variations-master/data"

# Reading the vectors from the given csv files
train1_90 = pd.read_csv(data_path+'/train1_90.csv')
train1_180 = pd.read_csv(data_path+'/train1_180.csv')
train1_360 = pd.read_csv(data_path+'/train1_360.csv')

train2_90 = pd.read_csv(data_path+'/train2_90.csv')
train2_180 = pd.read_csv(data_path+'/train2_180.csv')
train2_360 = pd.read_csv(data_path+'/train2_360.csv')

test_90 = pd.read_csv(data_path+'/test_90.csv')
test_180 = pd.read_csv(data_path+'/test_180.csv')
test_360 = pd.read_csv(data_path+'/test_360.csv')

def measure(a, b):
    mean_a, mean_b = np.mean(a), np.mean(b)
    std_a, std_b = np.std(a), np.std(b)
    n, result = len(a), 0
    for i in range(0,n):
        x = ((mean_a - a[i]) * (mean_b - b[i]))
        y = float(std_a * std_b * n)
        result += (x/y)
    return result

def computeDelta(wt, X, Xi):
    numerator, denominator = 0, 0
    n = len(X) - 1
    matrix = Xi.as_matrix()
    for i in range(0,len(matrix)):
        numerator +=  (matrix[i][n] * math.exp(wt * measure(X[0:n], matrix[i][0:n])))
        denominator += math.exp(wt * measure(X[0:n], matrix[i][0:n]))
    E_emp = float(numerator)/denominator
    return E_emp



# Bayesian Regression to predict the average price change for each dataset of train2 using train1 as input.
# Election of the weights is arbitrary
weight = 2
trainDeltaP90 = np.empty(0)
trainDeltaP180 = np.empty(0)
trainDeltaP360 = np.empty(0)
for i in range(0,len(train1_90.index)) :
  trainDeltaP90 = np.append(trainDeltaP90, computeDelta(weight,train2_90.iloc[i],train1_90))
for i in range(0,len(train1_180.index)) :
  trainDeltaP180 = np.append(trainDeltaP180, computeDelta(weight,train2_180.iloc[i],train1_180))
for i in range(0,len(train1_360.index)) :
  trainDeltaP360 = np.append(trainDeltaP360, computeDelta(weight,train2_360.iloc[i],train1_360))


# Actual deltaP values for the train2 data.
trainDeltaP = np.asarray(train2_360[['Yi']])
trainDeltaP = np.reshape(trainDeltaP, -1)


# Combine all the training data
d = {'deltaP': trainDeltaP,
     'deltaP90': trainDeltaP90,
     'deltaP180': trainDeltaP180,
     'deltaP360': trainDeltaP360 }
trainData = pd.DataFrame(d)


# Feed the data: [deltaP, deltaP90, deltaP180, deltaP360] to train the linear model / statsmodels ols function.
model = smf.ols(formula = 'deltaP ~ deltaP90 + deltaP180 + deltaP360', data = trainData).fit()
# Print the weights from the model
print(model.params)

# Perform the Bayesian Regression to predict the average price change for each dataset of test using train1 as input.
weight = 2
testDeltaP90 = testDeltaP180 = testDeltaP360 = np.empty(0)
for i in range(0,len(train1_90.index)) :
  testDeltaP90 = np.append(testDeltaP90, computeDelta(weight,test_90.iloc[i],train1_90))
for i in range(0,len(train1_180.index)) :
  testDeltaP180 = np.append(testDeltaP180, computeDelta(weight,test_180.iloc[i],train1_180))
for i in range(0,len(train1_360.index)) :
  testDeltaP360 = np.append(testDeltaP360, computeDelta(weight,test_360.iloc[i],train1_360))

# Actual deltaP values for test data.
testDeltaP = np.asarray(test_360[['Yi']])
testDeltaP = np.reshape(testDeltaP, -1)

# Test data
d = {'deltaP': testDeltaP,
     'deltaP90': testDeltaP90,
     'deltaP180': testDeltaP180,
     'deltaP360': testDeltaP360}
testData = pd.DataFrame(d)

# Prediction
result = model.predict(testData)

date = pd.date_range('10/10/2017', periods=50, freq='20S')

compare = { 'Actual': testDeltaP,
            'Predicted': result }
compareDF = pd.DataFrame(compare)

plt.plot(date,compareDF['Actual'])
plt.plot(date,compareDF['Predicted'])
plt.title("Prediction accuracy")
plt.xlabel("Time ")
plt.ylabel("Price difference (return)")

plt.show()

# MSE
MSE = 0.0
MSE = sm.mean_squared_error(compareDF['Actual'],compareDF['Predicted'])
print("The MSE is",MSE)
