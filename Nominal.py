import numpy as np
from Parameters import *
from Dynamics import *
from scipy.io import savemat
from scipy.io import loadmat
from Discretize import *
from numpy import dot

N = params['N']
M = params['M']
NUM = params['NUM']

tspan = params['t']
x0 = [0, 0, params['h0'], 0, 0, 0, params['m0']]
Xnom = odeint(EOM, x0, tspan, args=(params,), rtol=1e-13, atol=1e-13)
Unom = np.column_stack([params['K'] * Xnom[:, -1] * params['g'], np.zeros(len(tspan)), np.zeros(len(tspan))])
Xnom = Xnom.tolist()
Unom = Unom.tolist()
Ak, Bk, Ck, Sk, zk = DiscretizeAll(Xnom, Unom, params);

savemat('Nominal.mat', {'Xnom': Xnom, 'Unom': Unom, 'Ak': Ak, 'Bk': Bk,'Ck': Ck, 'Sk': Sk,'zk': zk})
'''
data = loadmat('Nominal.mat')
Xnom = data['Xnom']
Unom = data['Unom']
Ak = data['Ak']
Bk = data['Bk']
Ck = data['Ck']
Sk = data['Sk']
zk = data['zk']

for i in range(0, params['NUM']-1):
    should_be_zero = (Xnom[i+1, :].T - dot(Ak[:, :, i], Xnom[i, :].T) - dot(Bk[:, :, i], Unom[i, :].T) - dot(Ck[:, :,i], Unom[i+1, :].T)).reshape(-1, 1) - Sk[:,:, i]*params['sigma'] - zk[:,:, i]
    print(should_be_zero)
    #break;
    #print((- dot(Ak[:, :, i], Xnom[i, :].T) - dot(Bk[:, :, i], Unom[i, :].T) - dot(Ck[:, :,i], Unom[i+1, :].T) ).shape)
    #print((- Sk[:,:, i]*params['sigma'] - zk[:,:, i]).shape)
    #print((Xnom[i+1, :].T - dot(Ak[:, :, i], Xnom[i, :].T) - dot(Bk[:, :, i], Unom[i, :].T) - dot(Ck[:, :,i], Unom[i+1, :].T)).reshape(-1, 1).shape)
    
#data = loadmat('Nominal.mat')
#Bk = data['Bk']
#print(Bk[:, :, 23])
'''