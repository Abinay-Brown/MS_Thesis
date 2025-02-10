import numpy as np
from Linearize import *
from Dynamics import *
from scipy.interpolate import interp1d
import cvxpy as cvx
from cvxopt import matrix, solvers
from scipy.integrate import ode
from scipy.linalg import expm
from control import lqr
from numpy import pi
data = np.load('Nominal_Trajectory.npz')


Xnom = data['Xnom']
Unom = data['Unom']
# Reference Trajectory
xref = interp1d(params['t'], Xnom[0, :], fill_value="extrapolate")
yref = interp1d(params['t'], Xnom[1, :],  fill_value="extrapolate")
zref = interp1d(params['t'], Xnom[2, :],  fill_value="extrapolate")
vxref = interp1d(params['t'], Xnom[3, :],  fill_value="extrapolate")
vyref = interp1d(params['t'], Xnom[4, :],  fill_value="extrapolate")
vzref = interp1d(params['t'], Xnom[5, :],  fill_value="extrapolate")
mref = interp1d(params['t'], Xnom[6, :],  fill_value="extrapolate")
Tref = interp1d(params['t'], Unom[0, :],  fill_value="extrapolate")
aref = interp1d(params['t'], Unom[1, :],  fill_value="extrapolate")
bref = interp1d(params['t'], Unom[2, :],  fill_value="extrapolate")

control_list = [[Tref(0), aref(0), bref(0)]]
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
    Ref = np.array([xref(t), yref(t), zref(t), vxref(t), vyref(t), vzref(t), mref(t)])
    
    Ref_next = np.array([xref(t+dt), yref(t+dt), zref(t+dt), vxref(t+dt), vyref(t+dt), vzref(t+dt), mref(t+dt)])
    Uref = np.array([Tref(t), aref(t), bref(t)])
    X = np.array([x, y, z, xd, yd, zd, m])
    Q = 1000*np.eye(7)
    R = 0.00001*np.eye(3)
    parameters= [params['g'], params['Isp']]
    A, B = Linearize_guv(Ref, Uref, parameters)
    Ad, Bd = Discretize_guv_zoh(A, B, dt)
    K, S, E = lqr(A, B, Q, R)
    Uff = np.linalg.pinv(B) @ (Ref_next.T - A @ Ref.T)
    
    u = -K @ (X.T - Ref.T)
    
    #U = Uff - K @ (X.T - Ref.T)

    #print(K.shape)
    T = np.clip( Tref(t) + u[0], params['Thrust_lim'][0], params['Thrust_lim'][1])
    alpha = np.clip(aref(t) + u[1], params['alpha_lim'][0] *(pi/180),params['alpha_lim'][1] *(pi/180))
    beta =  np.clip(bref(t)  + u[2], params['beta_lim'][0] *(pi/180), params['beta_lim'][1] *(pi/180))
    #T = np.clip( U[0], params['Thrust_lim'][0], params['Thrust_lim'][1])
    #alpha = np.clip(U[1], params['alpha_lim'][0] *(pi/180),params['alpha_lim'][1] *(pi/180))
    #beta =  np.clip(U[2], params['beta_lim'][0] *(pi/180), params['beta_lim'][1] *(pi/180))
    
    control_list.append([T, alpha, beta])
    #print([T, alpha, beta])
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
sol, tsol = integrate_rk4(dynamics, 0, 59, Xnom[:, 0].T, params['inner_loop_dt'])
fig, ax = plt.subplots(3, 1, figsize=(8, 6))


# Plot on each subplot
ax[0].plot(tsol, sol[:, 0], label='LQR', color='r')
ax[0].plot(params['t'], Xnom[0, :], label='SCvx', color='b')

ax[0].set_ylabel("x (m)")
ax[0].legend()
ax[0].grid()

ax[1].plot(tsol, sol[:, 1], label='LQR', color='r')
ax[1].plot(params['t'], Xnom[1, :], label='SCvx', color='b')
ax[1].set_ylabel("y (m)")
ax[1].legend()
ax[1].grid()

ax[2].plot(tsol, sol[:, 2], label='LQR', color='g')
ax[2].plot(params['t'], Xnom[2, :], label='SCvx', color='b')
ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("z (m)")
ax[2].legend()
ax[2].grid()

# Adjust layout
plt.tight_layout()
plt.show()
