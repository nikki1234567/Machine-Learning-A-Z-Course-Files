#Apriori

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
#Training Apriori on dataset
