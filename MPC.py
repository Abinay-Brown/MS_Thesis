import numpy as np
from Linearize import *
from Dynamics import *
from scipy.interpolate import interp1d
import cvxpy as cvx
from cvxopt import matrix, solvers
from scipy.integrate import ode
from control import lqr
from numpy import pi
data = np.load('Nominal_Trajectory.npz')


Xnom = data['Xnom']
Unom = data['Unom']
# Reference Trajectory
xref = interp1d(params['t'], Xnom[0, :], kind='slinear', fill_value="extrapolate")
yref = interp1d(params['t'], Xnom[1, :], kind='slinear',  fill_value="extrapolate")
zref = interp1d(params['t'], Xnom[2, :],  kind='slinear', fill_value="extrapolate")
vxref = interp1d(params['t'], Xnom[3, :], kind='slinear',  fill_value="extrapolate")
vyref = interp1d(params['t'], Xnom[4, :],  kind='slinear', fill_value="extrapolate")
vzref = interp1d(params['t'], Xnom[5, :],  kind='slinear', fill_value="extrapolate")
mref = interp1d(params['t'], Xnom[6, :],  kind='slinear', fill_value="extrapolate")
Tref = interp1d(params['t'], Unom[0, :],  kind='slinear', fill_value="extrapolate")
aref = interp1d(params['t'], Unom[1, :],  kind='slinear', fill_value="extrapolate")
bref = interp1d(params['t'], Unom[2, :],  kind='slinear', fill_value="extrapolate")


u0 = cvx.Variable((3, 1))
u1 = cvx.Variable((3, 1))
u2 = cvx.Variable((3, 1))
u3 = cvx.Variable((3, 1))
u4 = cvx.Variable((3, 1))


x0 = cvx.Parameter((7, 1))
x1 = cvx.Variable((7, 1))
x2 = cvx.Variable((7, 1))
x3 = cvx.Variable((7, 1))
x4 = cvx.Variable((7, 1))
x5 = cvx.Variable((7, 1))


A0 = cvx.Parameter((7, 7))
A1 = cvx.Parameter((7, 7))
A2 = cvx.Parameter((7, 7))
A3 = cvx.Parameter((7, 7))
A4 = cvx.Parameter((7, 7))

B0 = cvx.Parameter((7 , 3))
B1 = cvx.Parameter((7 , 3))
B2 = cvx.Parameter((7 , 3))
B3 = cvx.Parameter((7 , 3))
B4 = cvx.Parameter((7 , 3))

R1 = cvx.Parameter((7, 1))
R2 = cvx.Parameter((7, 1))
R3 = cvx.Parameter((7, 1))
R4 = cvx.Parameter((7, 1))
R5 = cvx.Parameter((7, 1))

constraints = []

# Dynamics Contstraints
constraints += [x1 == A0 @ x0 + B0 @ u0,
                x2 == A1 @ x1 + B1 @ u1,
                x3 == A2 @ x2 + B2 @ u2,
                x4 == A3 @ x3 + B3 @ u3,
                x5 == A4 @ x4 + B4 @ u4]
# Control Constraints
constraints += [u0[0] >= params['Thrust_lim'][0], u0[0] <= params['Thrust_lim'][1], 
                u0[1] >= params['alpha_lim'][0] * (pi/180), u0[1] <= params['alpha_lim'][1]* (pi/180), 
                u0[2] >= params['beta_lim'][0]* (pi/180), u0[2] <= params['beta_lim'][1]* (pi/180)]
constraints += [u1[0] >= params['Thrust_lim'][0], u1[0] <= params['Thrust_lim'][1], 
                u1[1] >= params['alpha_lim'][0] * (pi/180), u1[1] <= params['alpha_lim'][1]* (pi/180), 
                u1[2] >= params['beta_lim'][0]* (pi/180), u1[2] <= params['beta_lim'][1]* (pi/180)]
constraints += [u2[0] >= params['Thrust_lim'][0], u2[0] <= params['Thrust_lim'][1], 
                u2[1] >= params['alpha_lim'][0] * (pi/180), u2[1] <= params['alpha_lim'][1]* (pi/180), 
                u2[2] >= params['beta_lim'][0]* (pi/180), u2[2] <= params['beta_lim'][1]* (pi/180)]
constraints += [u3[0] >= params['Thrust_lim'][0], u3[0] <= params['Thrust_lim'][1], 
                u3[1] >= params['alpha_lim'][0] * (pi/180), u3[1] <= params['alpha_lim'][1]* (pi/180), 
                u3[2] >= params['beta_lim'][0]* (pi/180), u3[2] <= params['beta_lim'][1]* (pi/180)]
constraints += [u4[0] >= params['Thrust_lim'][0], u4[0] <= params['Thrust_lim'][1], 
                u4[1] >= params['alpha_lim'][0] * (pi/180), u4[1] <= params['alpha_lim'][1]* (pi/180), 
                u4[2] >= params['beta_lim'][0]* (pi/180), u4[2] <= params['beta_lim'][1]* (pi/180)]

cost = cvx.norm(R1-x1) + cvx.norm(R2-x2) + cvx.norm(R3-x3) + cvx.norm(R4-x4) + cvx.norm(R5-x5)
#cost = cvx.norm(R1[2] - x1[2], 2) + cvx.norm(R2[2] - x2[2], 2) + cvx.norm(R3[2] - x3[2], 2) + cvx.norm(R4[2] - x4[2], 2) + cvx.norm(R5[2] - x5[2], 2)
cost = cvx.sum_squares(R1 - x1) + cvx.sum_squares(R2 - x2) + cvx.sum_squares(R3 - x3) + cvx.sum_squares(R4 - x4) + cvx.sum_squares(R5 - x5)

J = cvx.Minimize(cost)
prob = cvx.Problem(objective= J, constraints =  constraints)    

def mpc_control(t, dt, X0):
    # MPC Control input calculation
    r = np.array([[xref(t + dt), yref(t + dt), zref(t + dt), vxref(t + dt), vyref(t + dt), vzref(t + dt), mref(t + dt)],
                   [xref(t + 2*dt), yref(t + 2*dt), zref(t + 2*dt), vxref(t + 2*dt), vyref(t + 2*dt), vzref(t + 2*dt), mref(t + 2*dt)],
                   [xref(t + 3*dt), yref(t + 3*dt), zref(t + 3*dt), vxref(t + 3*dt), vyref(t + 3*dt), vzref(t + 3*dt), mref(t + 3*dt)],
                   [xref(t + 4*dt), yref(t + 4*dt), zref(t + 4*dt), vxref(t + 4*dt), vyref(t + 4*dt), vzref(t + 4*dt), mref(t + 4*dt)],
                   [xref(t + 5*dt), yref(t + 5*dt), zref(t + 5*dt), vxref(t + 5*dt), vyref(t + 5*dt), vzref(t + 5*dt), mref(t + 5*dt)]])
    u = np.array([[Tref(t + dt), aref(t + dt), bref(t + dt)],
                   [Tref(t + 2*dt), aref(t + 2*dt), bref(t + 2*dt)],
                   [Tref(t + 3*dt), aref(t + 3*dt), bref(t + 3*dt)],
                   [Tref(t + 4*dt), aref(t + 4*dt), bref(t + 4*dt)],
                   [Tref(t + 5*dt), aref(t + 5*dt), bref(t + 5*dt)]])
    
    parameters = [params['g'], params['Isp']]
    A, B = Linearize_guv(r, u, parameters)
    Ad, Bd = Discretize_guv_zoh(A, B, dt)
    
    x0.value = X0.reshape(7, 1)
    A0.value = Ad[0].reshape(7, 7)
    B0.value = Bd[0].reshape(7, 3)
    A1.value = Ad[1].reshape(7, 7)
    B1.value = Bd[1].reshape(7, 3)
    A2.value = Ad[2].reshape(7, 7)
    B2.value = Bd[2].reshape(7, 3)
    A3.value = Ad[3].reshape(7, 7)
    B3.value = Bd[3].reshape(7, 3)
    A4.value = Ad[4].reshape(7, 7)
    B4.value = Bd[4].reshape(7, 3)
    
    R1.value = r[0].reshape(7, 1)
    R2.value = r[1].reshape(7, 1)
    R3.value = r[2].reshape(7, 1)
    R4.value = r[3].reshape(7, 1)
    R5.value = r[4].reshape(7, 1) 
    error = prob.solve(verbose=False, solver = cvx.ECOS, warm_start = True)
    
    return u0.value
    #Q = np.diag([200, 200, 300, 50, 50, 50, 0])

def dynamics(t, state, control):
    g = params['g']
    Isp = params['Isp']
    dt = params['inner_loop_dt']
    
    # State Variables
    x, y, z = state[0], state[1], state[2]
    xd, yd, zd = state[3], state[4], state[5]
    m = state[6]
    
    # Control Variables
    #T = control[0]
    #alpha = control[1]
    #beta = control[2]
    
    X = np.array([x, y, z, xd, yd, zd, m])
    u = mpc_control(t, dt, X)
    #print(u)
    #print(K.shape)
    T = np.clip(u[0, 0], params['Thrust_lim'][0],params['Thrust_lim'][1])
    alpha = np.clip(u[1, 0], params['alpha_lim'][0] *(pi/180),params['alpha_lim'][1] *(pi/180))
    beta =  np.clip(u[2, 0], params['beta_lim'][0] *(pi/180), params['beta_lim'][1] *(pi/180))
    
    ax = -(T / m) * cos(alpha) * sin(beta)
    ay = (T / m) * sin(alpha)
    az = (T / m) * cos(alpha) * cos(beta) - g
    mdot = -T / (g * Isp)

    statedot = np.array([xd, yd, zd, ax, ay, az, mdot])
    
    return statedot  

def rk4_step(f, t, state, control, dt):
    k1 = f(t, state, control)
    k2 = f(t + 0.5 * dt, state + 0.5 * dt * k1, control)
    k3 = f(t + 0.5 * dt, state + 0.5 * dt * k2, control)
    k4 = f(t + dt, state + dt * k3, control)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def integrate_rk4(f, t0, tf, state0, dt):
    t, state = t0, np.array(state0)
    trajectory = [state]
    tsol = [t]
    while t < tf:
        state = rk4_step(f, t, state, None, dt)
        t += dt
        trajectory.append(state)
        tsol.append(t)
        print([t, state[0:3]])
        
    return np.array(trajectory), tsol

T_final =60
sol, tsol = integrate_rk4(dynamics, 0, 59, Xnom[:, 0].T, 1)
fig, ax = plt.subplots(3, 1, figsize=(8, 6))

# Plot on each subplot
ax[0].plot(tsol, sol[:, 0], label='MPC', color='r')
ax[0].plot(params['t'], Xnom[0, :], label='SCvx', color='b')
ax[0].plot(tsol, xref(tsol), label='interp', color='g')

ax[0].set_ylabel("x (m)")
ax[0].legend()
ax[0].grid()

ax[1].plot(tsol, sol[:, 1], label='MPC', color='r')
ax[1].plot(params['t'], Xnom[1, :], label='SCvx', color='b')
ax[1].plot(tsol, yref(tsol), label='interp', color='g')

ax[1].set_ylabel("y (m)")
ax[1].legend()
ax[1].grid()

ax[2].plot(tsol, sol[:, 2], label='MPC', color='r')
ax[2].plot(params['t'], Xnom[2, :], label='SCvx', color='b')
ax[2].plot(tsol, zref(tsol), label='interp', color='g')
ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("z (m)")
ax[2].legend()
ax[2].grid()

# Adjust layout
plt.tight_layout()
plt.show()
