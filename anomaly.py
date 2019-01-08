import random
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


MIN = 1
MAX = 10

data = np.random.randint(low = MIN*100, high = MAX * 100 + 1, size = (100, 5))
dataf = np.array(data, dtype = np.float64)/100

#avreage on rows 
datamean = np.mean(dataf,axis = 0)
datastd = np.std(dataf, axis = 0) 


pdist = norm(datamean,datastd)

# this contains all the probabilities in the dataset to belong to the given probability 
pds  = pdist.pdf(dataf)

# prod on columns 
epsilons = np.prod ( pds, axis = 1 )

plt.plot(epsilons)
plt.ylabel('epsilon')
plt.xlabel('sample index')
plt.show()






