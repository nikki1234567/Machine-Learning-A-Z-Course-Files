# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
import math
import random
N=10000
d=10
ads_selected=[]
total_reward=0
numbers_of_rewards1=[0] * d
numbers_of_rewards0=[0] * d
for n in range(0,N):
    ad=0
    max_random=0
    for i in range(0,d):
        random_beta=random.betavariate(numbers_of_rewards1[i]+1,numbers_of_rewards0[i]+1)
        
        if max_random < random_beta:
            max_random=random_beta
            ad=i
            
    ads_selected.append(ad)
    reward=dataset.values[n,ad]
    if reward==1:
        numbers_of_rewards1[ad]+=1
    else:
        numbers_of_rewards0[ad]+=1
    total_reward+=reward

#Visualizing results

plt.hist(ads_selected)
plt.title('Histogram of Ad selections')
plt.xlabel('Ads')
plt.ylabel('Number of times Ad was selected')
plt.show()



