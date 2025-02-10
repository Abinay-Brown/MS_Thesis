import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from Parameters import *
from numpy import sin, cos, tan, pi
from Integrators import rk8_step
import matplotlib.pyplot as plt

# Load Reference Trajectory
data = np.load('Nominal_Trajectory.npz')

Xnom = data['Xnom']
Unom = data['Unom']
xr = interp1d(params['t'], Xnom[0, :], fill_value="extrapolate")
yr = interp1d(params['t'], Xnom[1, :], fill_value="extrapolate")
zr = interp1d(params['t'], Xnom[2, :], fill_value="extrapolate")
xdr = interp1d(params['t'], Xnom[3, :], fill_value="extrapolate")
ydr = interp1d(params['t'], Xnom[4, :], fill_value="extrapolate")
zdr = interp1d(params['t'], Xnom[5, :], fill_value="extrapolate")

xddr = interp1d(params['t'], np.gradient(Xnom[3, :], params['t']),   fill_value="extrapolate")
yddr = interp1d(params['t'], np.gradient(Xnom[4, :], params['t']), fill_value="extrapolate")
zddr = interp1d(params['t'], np.gradient(Xnom[5, :], params['t']), fill_value="extrapolate")

def lyapunov_control(t, X, xref):
    x, y, z, xd, yd, zd, m = X
    g = params['g']
    #print(X) 
    # Positive Coefficients
    kx, ky, kz = 0.0001, 0.0001, 0.0001
    kvx, kvy, kvz = 1, 1, 2
    
    # Position Errors
    ex = x - xref[0]
    ey = y - xref[1]
    ez = z - xref[2]
    
    # Velocity Errors
    exd = xd - xref[3]
    eyd = yd - xref[4]
    ezd = zd - xref[5]
    if exd == 0:
        exd = 0.0000001
    if eyd == 0:
        eyd = 0.0000001
    if ezd == 0:
        ezd = 0.0000001
    axd = xref[6] - ex - (kx*ex*ex/exd) - kvx * exd
    ayd = xref[7] - ey - (ky*ey*ey/eyd) - kvy * eyd
    azd = xref[8] - ez - (kz*ez*ez/ezd) - kvz * ezd + g
    
    amag = np.sqrt(axd*axd + ayd*ayd + azd*azd)
    T = m*amag
    T = np.clip(T, params['Thrust_lim'][0], params['Thrust_lim'][1])
    # Normalize acceleration vector
    axd /= amag
    ayd /= amag
    azd /= amag
    
    tan_beta = - axd / azd
    tan_beta = np.clip(tan_beta, tan(params['beta_lim'][0]*pi/180), tan(params['beta_lim'][1]*pi/180))
    beta = np.arctan(tan_beta)
    sin_alpha = ayd
    sin_alpha = np.clip(sin_alpha, sin(params['alpha_lim'][0]*pi/180), sin(params['alpha_lim'][1]*pi/180))
    alpha = np.arcsin(sin_alpha)
    u = [T, alpha, beta]
    
    return u

'''
def integrate_rk4(f, t0, tf, state0, dt):
    t, state = t0, np.array(state0)
    trajectory = [state]
    tsol = [t]
    while t < tf:
        ctrl = lyapunov_control(t, state)
        state = rk8_step(f, t, state, dt, ctrl)
        t += dt
        trajectory.append(state)
        tsol.append(t)
        print([t, state[0:3]])
        
    return np.array(trajectory), tsol
T_final =60
sol, tsol = integrate_rk4(dynamics, 0, T_final, Xnom[:, 0].T, params['inner_loop_dt'])

fig, ax = plt.subplots(4, 1, figsize=(8, 6))


# Plot on each subplot
ax[0].plot(tsol, sol[:, 0], label='Lyapunov', color='r')
ax[0].plot(params['t'], Xnom[0, :], label='SCvx', color='b')
#ax[0].plot(tsol, xr(tsol), label='interp', color='r')
ax[0].set_ylabel("x (m)")
ax[0].legend()
ax[0].grid()

ax[1].plot(tsol, sol[:, 1], label='Lyapunov', color='r')
ax[1].plot(params['t'], Xnom[1, :], label='SCvx', color='b')
ax[1].set_ylabel("y (m)")
ax[1].legend()
ax[1].grid()

ax[2].plot(tsol, sol[:, 2], label='Lyapunov', color='r')
ax[2].plot(params['t'], Xnom[2, :], label='SCvx', color='b')
ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("z (m)")
ax[2].legend()
ax[2].grid()


#ax[3].plot(tsol, sol[:, 4], label='Lyapunov', color='r')
ax[3].plot(params['t'], Unom[0, :], label='SCvx', color='b')
ax[3].set_xlabel("Time (s)")
ax[3].set_ylabel("m (kg)")
ax[3].legend()
ax[3].grid()
# Adjust layout
plt.tight_layout()
plt.show()
'''