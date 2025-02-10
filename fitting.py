import numpy as np
from numpy import sin, cos
from Parameters import *
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
data = np.load('Nominal_Trajectory.npz')


Xnom = data['Xnom']
Unom = data['Unom']
# Reference Trajectory
Tref = interp1d(params['t'], Unom[0, :],  kind='slinear', fill_value="extrapolate")
aref = interp1d(params['t'], Unom[1, :],  kind='slinear', fill_value="extrapolate")
bref = interp1d(params['t'], Unom[2, :],  kind='slinear', fill_value="extrapolate")

axf = interp1d(params['t'], np.gradient(Xnom[3, :], params['t']), kind='cubic',   fill_value="extrapolate")
ayf = interp1d(params['t'], np.gradient(Xnom[4, :], params['t']), kind='cubic', fill_value="extrapolate")
azf = interp1d(params['t'], np.gradient(Xnom[5, :], params['t']), kind='cubic', fill_value="extrapolate")
mdotf = interp1d(params['t'], np.gradient(Xnom[6, :], params['t']), kind='cubic', fill_value="extrapolate")
mref = interp1d(params['t'], Xnom[6, :],  fill_value="extrapolate")
#plt.plot(tspan, mref(tspan))
#plt.show()

g = params['g']
Isp = params['Isp']

def iterative_lumve2(y, x, m, max_iter=10, tol=1e-3):
    """
    Iteratively refines T, alpha, beta using LUMVE (Linearized Update via Matrix Vector Equation)
    with the Newton update equation.
    """
    # Initialize variables
    T, alpha, beta = x
    Isp = params['Isp']
    g = params['g']
    
    for i in range(max_iter):
        # Compute Jacobian matrix H at current estimate
        H = np.array([
            [-cos(alpha) * sin(beta) / m, (T/m) * sin(alpha) * sin(beta), - (T/m) * cos(alpha) * cos(beta)],
            [-sin(alpha) / m, - (T/m) * cos(alpha), 0],
            [-cos(alpha) * cos(beta) / m, (T/m) * sin(alpha) * cos(beta), (T/m) * cos(alpha) * sin(beta)],
        ])
        
        # Compute residual error and solve for update step
        delta_x = np.linalg.lstsq(H, y - H @ x, rcond=None)[0]
        x_new = x + delta_x

        # Extract new estimates
        T_new, alpha_new, beta_new = x_new
        
        # Enforce constraints
        T_new = np.clip(T_new, 1, 1500 * 1.625)  # Ensure thrust is within limits
        alpha_new = np.clip(alpha_new, np.radians(-30), np.radians(30))  
        beta_new = np.clip(beta_new, np.radians(-30), np.radians(30))  
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            break
        
        # Update for next iteration
        T, alpha, beta = T_new, alpha_new, beta_new
        x = np.array([T, alpha, beta])

    return T, alpha, beta


tspan = np.linspace(0, 60, 200)
Uopt = np.zeros((3, 200))
for i in range(len(tspan)):
    time_stamp = tspan[i]
    y = np.array([axf(time_stamp), ayf(time_stamp), azf(time_stamp)]).T
    x = np.array([Tref(time_stamp), aref(time_stamp), bref(time_stamp)]).T
    T2, a2, b2  = iterative_lumve2(y, x, mref(tspan[i]))
    Uopt[0, i] = T2
    Uopt[1, i] = a2
    Uopt[2, i] = b2
    print(i)
    #print([T2, a2, b2 , m2])
    #print([Tref(time_stamp), np.degrees(aref(time_stamp)), np.degrees(bref(time_stamp)), mref(time_stamp)])

Tref2 = interp1d(tspan, Uopt[0, :],  fill_value="extrapolate")
aref2 = interp1d(tspan, Uopt[1, :],  fill_value="extrapolate")
bref2 = interp1d(tspan, Uopt[2, :],  fill_value="extrapolate")

def accelerometer(t, state, control):
    g = 1.625
    
    # State Variables
    x, y, z = state[0], state[1], state[2]
    xd, yd, zd = state[3], state[4], state[5]
    m = state[6]
    Isp = 350
    T = Tref2(t)
    alpha = aref2(t)
    beta = bref2(t)
    ax = -(T / m) * cos(alpha) * sin(beta)
    ay = (T / m) * sin(alpha)
    az = (T / m) * cos(alpha) * cos(beta) - g
    mdot = -T / (g * Isp)
    #ax = axf(t)
    #ay = ayf(t)
    #az = azf(t)
    #mdot = mdotf(t)
    print([ax, axf(t)])
    statedot = np.array([xd, yd, zd, ax, ay, az, mdot])
    #print([ax, axf(t)])
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
        #print([t, state[0:3]])
        
    return np.array(trajectory), tsol
T_final =60
sol, tsol = integrate_rk4(accelerometer, 0, 59, Xnom[:, 0].T, params['inner_loop_dt'])
fig, ax = plt.subplots(3, 1, figsize=(8, 6))


# Plot on each subplot
ax[0].plot(tspan, sol[:, 0], label='LQR', color='r')
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