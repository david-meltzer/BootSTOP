import random
import numpy as np

def rho(z):
    return z/(1+np.sqrt((1-z)))**2

def sample_z(lambda_max,size):
    result=[]

    while len(result)<size:
        z=(random.random()+1)/2
        if rho(z)+rho(1-z)<lambda_max:
            result.append(z)
    return np.array(result)
