import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[:, 1:2].values
y= dataset.iloc[:, 2].values

#fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#fitting polynomial regression ro dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualizing Linear Regression results

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or bluff(Linear Regression)')
plt.xlabel('Positions')
plt.ylabel('Salary')
plt.show()

#Visualizing Polynomial Regression results
# X_grid=np.arange(min(X),max(X),0.1)
# X_grid=X_grid.reshape((len(X_grid),1))

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth or bluff(Linear Regression)')
plt.xlabel('Positions')
plt.ylabel('Salary')
plt.show()



print(lin_reg.predict([[6.5]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))
