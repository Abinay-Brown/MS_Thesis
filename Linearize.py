import numpy as np
from Parameters import *
from math import sin, cos
import math
from numba import guvectorize, float64, float32, int64
import time
import matplotlib.pyplot as plt
#from Dynamics import *

@guvectorize(['void(float64[:], float64[:], float64[:], float64[:,:], float64[:,:])'], 
             '(n),(m),(p)->(n,n),(n,m)', nopython=True, cache=True)
def Linearize_guv(X, U, params, A, B):
    g = params[0]  # Gravitational acceleration
    Isp = params[1]  # Specific impulse
    
    # Unpack State Variables
    x, y, z = X[0], X[1], X[2]
    vx, vy, vz = X[3], X[4], X[5]
    m = X[6]
    
    # Unpack Control Variables
    T = U[0]
    alpha = U[1]
    beta = U[2]
    
    # Reset A and B matrices to zero
    A[:, :] = 0.0  # Set all elements of A to 0
    B[:, :] = 0.0  # Set all elements of B to 0
    
    # Populate the A matrix
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0

    A[3, 6] = T * cos(alpha) * sin(beta) / m**2
    A[4, 6] = -T * sin(alpha) / m**2
    A[5, 6] = -T * cos(alpha) * cos(beta) / m**2

    # Populate the B matrix
    B[3, 0] = -cos(alpha) * sin(beta) / m
    B[3, 1] = T * sin(alpha) * sin(beta) / m
    B[3, 2] = -T * cos(alpha) * cos(beta) / m
    
    B[4, 0] = sin(alpha) / m
    B[4, 1] = T * cos(alpha) / m
    
    B[5, 0] = cos(alpha) * cos(beta) / m
    B[5, 1] = -T * cos(beta) * sin(alpha) / m
    B[5, 2] = -T * cos(alpha) * sin(beta) / m
    
    B[6, 0] = -1.0 / (Isp * g)
    

'''
First-Order-Hold Discretization
'''    
@guvectorize(['void(float64[:,:], float64[:,:], float64, float64[:,:], float64[:,:], float64[:,:])'], 
             '(n,m),(n,p),()->(n,m),(n,p),(n,p)', nopython=True, cache=True)
def Discretize_guv(A, B, dt, Ad, Bd, Cd):
    dim = 7  
    iter_count = 30  
    N = 500  
    dtau = dt / N  

    for i in range(dim):
        for j in range(dim):
            Ad[i, j] = 1.0 if i == j else 0.0

    

    Apow = np.eye(dim, dtype=np.float64)

    fact = 1.0  # Factorial
    for k in range(1, iter_count + 1):
        
        Apow_new = np.zeros((dim, dim), dtype=np.float64)
        for r in range(dim):
            for s in range(dim):
                for t in range(dim):
                    Apow_new[r, s] += Apow[r, t] * A[t, s]
        
        Apow = Apow_new.copy()
        fact *= k
        
        for r in range(dim):
            for s in range(dim):
                Ad[r, s] += (Apow[r, s] * dt) / fact


    # Initialize B1d and B2d as zero matrices
    B1d = np.zeros((dim, 3), dtype=np.float64)
    B2d = np.zeros((dim, 3), dtype=np.float64)
    
    for i in range(dim):
        for j in range(3):
            Bd[i, j] = 0
            Cd[i, j] = 0
            
    # Numerical integration to compute B1d and B2d
    for step in range(1, N + 1):
        tau = step * dtau
        Texp = dt - tau

        # Initialize expm as identity matrix
        expm = np.eye(dim, dtype=np.float64)
        Apow = np.eye(dim, dtype=np.float64)
        fact = 1.0

        # Compute expm using Taylor series
        for k in range(1, iter_count + 1):
            Apow_new = np.zeros((dim, dim), dtype=np.float64)
            for r in range(dim):
                for s in range(dim):
                    for t in range(dim):
                        Apow_new[r, s] += Apow[r, t] * A[t, s]
            Apow = Apow_new.copy()
            fact *= k
            for r in range(dim):
                for s in range(dim):
                    Apow[r, s] *= Texp / fact
            for r in range(dim):
                for s in range(dim):
                    expm[r, s] += Apow[r, s]

        # Multiply expm with B: expmB = expm * B
        expmB = np.zeros((dim, 3), dtype=np.float64)
        for i in range(dim):
            for j in range(3):
                for k in range(dim):
                    expmB[i, j] += expm[i, k] * B[k, j]

        # Add the scaled contribution to B1d and B2d
        for i in range(dim):
            for j in range(3):
                B1d[i, j] += expmB[i, j] * dtau
                B2d[i, j] += expmB[i, j] * tau / dt * dtau

    for i in range(dim):
        for j in range(3):
            Bd[i, j] = B1d[i, j] - B2d[i, j]
            Cd[i, j] = B2d[i, j]
    # Print the result
    #print("B1d:")
    #for row in B1d:
    #    print(row)
    #print("B2d:")
    #for row in B2d:
    #    print(row)

@guvectorize(['void(float64[:,:], float64[:,:], float64, float64[:,:], float64[:,:])'], 
             '(n,m),(n,p),()->(n,m),(n,p)', nopython=True, cache=True)
def Discretize_guv_zoh(A, B, dt, Ad, Bd):
    dim = 7  
    iter_count = 30  
    N = 500  
    dtau = dt / N  

    for i in range(dim):
        for j in range(dim):
            Ad[i, j] = 1.0 if i == j else 0.0

    

    Apow = np.eye(dim, dtype=np.float64)

    fact = 1.0  # Factorial
    for k in range(1, iter_count + 1):
        
        Apow_new = np.zeros((dim, dim), dtype=np.float64)
        for r in range(dim):
            for s in range(dim):
                for t in range(dim):
                    Apow_new[r, s] += Apow[r, t] * A[t, s]
        
        Apow = Apow_new.copy()
        fact *= k
        
        for r in range(dim):
            for s in range(dim):
                Ad[r, s] += (Apow[r, s] * dt) / fact


    # Initialize B1d and B2d as zero matrices
    B1d = np.zeros((dim, 3), dtype=np.float64)
    B2d = np.zeros((dim, 3), dtype=np.float64)
    
    for i in range(dim):
        for j in range(3):
            Bd[i, j] = 0
            
    # Numerical integration to compute B1d and B2d
    for step in range(1, N + 1):
        tau = step * dtau
        Texp = dt - tau

        # Initialize expm as identity matrix
        expm = np.eye(dim, dtype=np.float64)
        Apow = np.eye(dim, dtype=np.float64)
        fact = 1.0

        # Compute expm using Taylor series
        for k in range(1, iter_count + 1):
            Apow_new = np.zeros((dim, dim), dtype=np.float64)
            for r in range(dim):
                for s in range(dim):
                    for t in range(dim):
                        Apow_new[r, s] += Apow[r, t] * A[t, s]
            Apow = Apow_new.copy()
            fact *= k
            for r in range(dim):
                for s in range(dim):
                    Apow[r, s] *= Texp / fact
            for r in range(dim):
                for s in range(dim):
                    expm[r, s] += Apow[r, s]

        # Multiply expm with B: expmB = expm * B
        expmB = np.zeros((dim, 3), dtype=np.float64)
        for i in range(dim):
            for j in range(3):
                for k in range(dim):
                    expmB[i, j] += expm[i, k] * B[k, j]

        # Add the scaled contribution to B1d and B2d
        for i in range(dim):
            for j in range(3):
                B1d[i, j] += expmB[i, j] * dtau
                

    for i in range(dim):
        for j in range(3):
            Bd[i, j] = B1d[i, j] 
            

def Initial_Guess():
    N = params['N']
    M = params['M']
    NUM = params['NUM']

    tspan = params['t']
    x0 = [40, 40, params['h0'], 0, 0, 0, params['m0']]
    Xnom = odeint(EOM, x0, tspan, args=(params,), rtol=1e-13, atol=1e-13)
    Unom = np.column_stack([params['K'] * Xnom[:, -1] * params['g'], np.zeros(len(tspan)), np.zeros(len(tspan))])
    
    return Xnom, Unom

'''
Xnom, Unom = Initial_Guess()
parameters = [params['g'], params['Isp']]
A, B = Linearize_guv(Xnom, Unom, parameters)
Ad, Bd, Cd = Discretize_guv(A, B, params['dt'])


Xsol = np.zeros((params['NUM'], 7))
Xsol[0, :] = Xnom[0, :]
tspan = params['t']
for i in range(params['NUM'] - 1):
    #print(i)
    X = Ad[i, :, :].dot(Xnom[i, :].T) + Bd[i, :, :].dot(Unom[i, :].T) + Cd[i, :, :].dot(Unom[i+1, :].T) 
    Xsol[i+1, :] = X.T

print(Xnom[-1, :])
print(Xsol[-1, :])

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Trajectory", fontsize=16)

ax = axes[0, 0]
ax.plot(tspan, Xsol[:, 0])
ax.plot(tspan, Xnom[:, 0])

ax.set_ylabel("X (m)")
ax.set_xlabel("time (sec)")

ax = axes[0, 1]
ax.plot(tspan, Xsol[:, 1])
ax.plot(tspan, Xnom[:, 1])

ax.set_ylabel("Y (m)")
ax.set_xlabel("time (sec)")

ax = axes[0, 2]
ax.plot(tspan, Xsol[:, 2])
ax.plot(tspan, Xnom[:, 2])
ax.set_ylabel("Z (m)")
ax.set_xlabel("time (sec)")

ax = axes[1, 0]
ax.plot(tspan, Xsol[:, 3])
ax.plot(tspan, Xnom[:, 3])
ax.set_ylabel("Vx (m/s)")
ax.set_xlabel("time (sec)")

ax = axes[1, 1]
ax.plot(tspan, Xsol[:, 4])
ax.plot(tspan, Xnom[:, 4])
ax.set_ylabel("Vy (m/s)")
ax.set_xlabel("time (sec)")

ax = axes[1, 2]
ax.plot(tspan, Xsol[:, 5])
ax.plot(tspan, Xnom[:, 5])
ax.set_ylabel("Vz (m/s)")
ax.set_xlabel("time (sec)")


ax = axes[0, 3]
ax.plot(tspan, Xsol[:, 6])
ax.plot(tspan, Xnom[:, 6])
ax.set_ylabel("Mass (kg)")
ax.set_xlabel("time (sec)")

ax = axes[0, 4]
ax.plot(tspan, Unom[:, 0])
ax.set_ylabel("Thrust (kg)")
ax.set_xlabel("time (sec)")


ax = axes[1, 3]
ax.plot(tspan, Unom[:, 1])
ax.set_ylabel("alpha (rad)")
ax.set_xlabel("time (sec)")

ax = axes[1, 4]
ax.plot(tspan, Unom[:, 2])
ax.set_ylabel("beta (rad)")
ax.set_xlabel("time (sec)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
'''
