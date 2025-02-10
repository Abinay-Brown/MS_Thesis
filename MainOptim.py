import cvxpy as cvx
from numpy import sin, cos, tan, eye, pi
from Parameters import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


''' Loading Nominal Trajectory '''
data = loadmat('Nominal.mat')
Xnom = data['Xnom']
Unom = data['Unom']
Ak = data['Ak']
Bk = data['Bk']
Ck = data['Ck']
Sk = data['Sk']
zk = data['zk']

# Extracting Parameters
N = params['N']
M = params['M']
NUM = params['NUM']
m0 = params['m0']
h0 = params['h0']
g = params['g']

# Boundary Conditions
xi = [0, 0, h0, 0, 0, 0, m0]
xf = [15.00, 20.00, 0, 0, 0, 0, m0-500]

''' Declaring Decision Variables '''
X = cvx.Variable((N, NUM)) # State Variables
U = cvx.Variable((M, NUM)) # Control Input Variables
NU = cvx.Variable((N, NUM - 1)) # Artificial Infeasibility
Sig = cvx.Variable(nonneg = True)

# Declaring Discrete State-Space Parameters
Ah = cvx.Parameter((N * N, NUM - 1))
Bh = cvx.Parameter((N * M, NUM - 1))
Ch = cvx.Parameter((N * M, NUM - 1))
Sh = cvx.Parameter((N * 1, NUM - 1))
zh = cvx.Parameter((N * 1, NUM - 1))

# Declaring Nominal Trajectory as Parameters
Xprev = cvx.Parameter((N, NUM))
Uprev = cvx.Parameter((M, NUM))
Sigprev = cvx.Parameter(nonneg = True)

# Weights and Trust Regions
Wsigma = cvx.Parameter(nonneg = True)
Wnu = cvx.Parameter(nonneg = True)
Delta = cvx.Parameter(nonneg = True)

''' Initializing Trajectory Guess '''
X.value = Xnom.T
U.value = Unom.T
NU.value = np.zeros((N, NUM - 1))
Sig.value = params['sigma']

''' Initializing Parameters '''
Ah.value = Ak
Bh.value = Bk
Ch.value = Ck
Sh.value = Sk
zh.value = zk
Xprev.value = Xnom.T
Uprev.value = Unom.T
Sigprev.value = params['sigma']
Wsigma.value = 0.1
Wnu.value = 10
Delta.value = 500

''' Adding Constraints '''
Constraints = []
# Boundary Conditions

Constraints += [ X[0, 0] == xi[0],
                 X[1, 0] == xi[1],
                 X[2, 0] == xi[2],
                 X[3, 0] == xi[3],
                 X[4, 0] == xi[4],
                 X[5, 0] == xi[5],
                 X[6, 0] == xi[6],
                 X[0, -1] == xf[0],
                 X[1, -1] == xf[1],
                 X[2, -1] == xf[2],
                 X[3, -1] == xf[3],
                 X[4, -1] == xf[4],
                 X[5, -1] == xf[5], 
                 X[6, -1] >= xf[6]] # lower final mass constraint

# Control Constraints
for i in range(0, NUM):
    Constraints += [U[0, i] <= 1.2*m0*g, U[1, i] <= pi/4, U[2, i] <= pi/4, U[0, i] >= 0, U[1, i] >= -pi/4, U[2, i] >= -pi/4]
 
for i in range(1, 5):
    Constraints += [X[3, -1 - i] == 0, X[4, -1 - i] == 0, X[5, -1 - i] <= -1]
for i in range(0, NUM):
    Constraints += [X[0, i] >=0, X[1, i] >= 0, X[2, i] >= 0]
    
# Dynamic Constraints
for i in range(0, NUM-1):
    Constraints += [X[:, i+1] == cvx.reshape(Ah[:, i], (N, N), order='C')@X[:, i]
                    +cvx.reshape(Bh[:, i], (N, M), order='C')@U[:, i]
                    +cvx.reshape(Ch[:, i], (N, M), order='C')@U[:, i+1]
                    + Sig*Sh[:, i] + zh[:, i] + NU[:, i]]

# Trust Regions Constraints
dX = X - Xprev
dU = U - Uprev
dS = Sig - Sigprev
Constraints += [cvx.norm(dX, 1) + cvx.norm(dU, 1) + cvx.norm(dS, 1) <= Delta]

Objective_Func = cvx.Minimize(Wsigma*Sig + Wnu * cvx.norm(NU, 1))

prob = cvx.Problem(objective= Objective_Func, constraints =  Constraints)
error = prob.solve(verbose=True, solver = cvx.ECOS)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the values (x, y, z)
ax.plot(X.value[0, :], X.value[1, :], X.value[2, :], label="3D Trajectory")
print(X.value[:, -1])
print(U.value[:, -1])
print(NU.value[:, -7])
print(Sig.value)
# Set labels
ax.set_xlabel('X axis (m)')
ax.set_ylabel('Y axis (m)')
ax.set_zlabel('Z axis (m)')
ax.set_title('Successive Convexification Optimal Trajectory')
plt.show()
'''

tspan = np.linspace(0, Sig.value, NUM)
plt.subplot(3, 2, 1);
plt.plot(tspan, X.value[0, :])
plt.xlabel(" time (sec) ")
plt.ylabel(" x (m) ")

plt.subplot(3, 2, 3);
plt.plot(tspan, X.value[1, :])
plt.xlabel(" time (sec) ")
plt.ylabel(" y (m) ")

plt.subplot(3, 2, 5);
plt.plot(tspan, X.value[2, :])
plt.xlabel(" time (sec) ")
plt.ylabel(" z (m) ")


plt.subplot(3, 2, 2);
plt.plot(tspan, X.value[3, :])
plt.xlabel(" time (sec) ")
plt.ylabel(" Vx (m/s) ")


plt.subplot(3, 2, 4);
plt.plot(tspan, X.value[4, :])
plt.xlabel(" time (sec) ")
plt.ylabel(" Vy (m/s) ")

plt.subplot(3, 2, 6);
plt.plot(tspan, X.value[5, :])
plt.xlabel(" time (sec) ")
plt.ylabel(" Vz (m/s) ")

plt.show()
'''
'''

tspan = np.linspace(0, Sig.value, NUM)
plt.subplot(3, 1, 1);
plt.plot(tspan, U.value[0, :])
plt.xlabel(" time (sec) ")
plt.ylabel(" Thrust (N) ")

plt.subplot(3, 1, 2);
plt.plot(tspan, U.value[1, :]*180/np.pi)
plt.xlabel(" time (sec) ")
plt.ylabel(" alpha (deg) ")

plt.subplot(3, 1, 3);
plt.plot(tspan, U.value[2, :]*180/np.pi)
plt.xlabel(" time (sec) ")
plt.ylabel(" beta (deg) ")
plt.show();
'''