import numpy as np
from Parameters import *
from Integrators import *
from Lyapunov import lyapunov_control
from Dynamics import dynamics
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from KalmanFilter import UKF_update, Sensor_Measurement


 
    
IC = [0, 0, params['h0'], 0, 0, 0, params['m0']]
FC = [15.00, 20.00, 0, 0, 0, 0, params['m0']-500]
GC = [40, 40, params['h0'], 0, 0, 0, params['m0']]
Hazards = [[0, 0, 20, 20], [30, 30, 20, 20]]
Sigx = np.array([[200, 0], [0, 500]])
#opt.test()
#opt.PlotTrajectory()

#np.savez('Nominal_Trajectory.npz', Xnom = opt.X.value, Unom = opt.U.value)

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



tspan = np.arange(0, params['T'], 1/params['control_freq'])
nominal = np.zeros((len(tspan), params['N']))
predict = np.zeros((len(tspan), params['N']))

state = np.array([0, 0, params['h0'], 0, 0, 0, params['m0']])
state_est = state
state_int = state[0:-1]

ind = 0
dt = 1/params['control_freq']

Pukf = ukf_params['Pukf']

for t in tspan:
    #predict[ind, :] = np.hstack((state_int, 0))
    nominal[ind, :] = state
    predict[ind, :] = state_est
    
    # Determine Control-Input based on state estimate
    ref_traj = np.array([xr(t), yr(t), zr(t), xdr(t), ydr(t), zdr(t), xddr(t), yddr(t), zddr(t)])
    ctrl = lyapunov_control(t, state_est, ref_traj)
    
    #print(state_est.shape)
    #Generate Noisy Accelerometer data
    statedot = dynamics(t, state, ctrl)
    state = rk4_step(dynamics, t, state, dt, ctrl)
    z, R, state_int = Sensor_Measurement(ind, state_int, statedot, state)
    state_est, Pukf = UKF_update(t, state_est, Pukf, z,  R, ctrl)
    
    ind += 1
    print(state_est[0:3])
    #break;
    #print(ind)
    

'''
fig, ax = plt.subplots(3, 1, figsize=(8, 6))


# Plot on each subplot
ax[0].plot(tspan, predict[:, 0], label='Lyapunov+UKF', color='r')
ax[0].plot(tspan, nominal[:, 0], label='True', color='g')
ax[0].plot(params['t'], Xnom[0, :], label='SCvx', color='b')
#ax[0].plot(tsol, xr(tsol), label='interp', color='r')
ax[0].set_ylabel("x (m)")
ax[0].legend()
ax[0].grid()

ax[1].plot(tspan, predict[:, 1], label='Lyapunov+UKF', color='r')
ax[1].plot(tspan, nominal[:, 1], label='True', color='g')
ax[1].plot(params['t'], Xnom[1, :], label='SCvx', color='b')
ax[1].set_ylabel("y (m)")
ax[1].legend()
ax[1].grid()

ax[2].plot(tspan, predict[:, 2], label='Lyapunov+UKF', color='r')
ax[2].plot(tspan, nominal[:, 2], label='True', color='g')
ax[2].plot(params['t'], Xnom[2, :], label='SCvx', color='b')
ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("z (m)")
ax[2].legend()
ax[2].grid()


plt.tight_layout()
plt.show()
'''

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(Xnom[0, :], Xnom[1, :], Xnom[2, :], label="SCvx", color='r')
cx = Xnom[0, -1]
cy = Xnom[1, -1]
a = np.sqrt(Sigx[0, 0])
b = np.sqrt(Sigx[1, 1])
theta = np.linspace(0, 2 * np.pi, 100)  # Angle parameter for the ellipse
x = a * np.cos(theta)  # Parametric x-coordinates
y = b * np.sin(theta)  # Parametric y-coordinates
z = np.zeros_like(x)  # Ellipse is on the xy-plane (z=0)
# Translate the ellipse to its center (cx, cy)
x_centered = x + cx
y_centered = y + cy
ax.plot(x_centered, y_centered, z, label=f"99% Confidence Zone ({np.round(cx, 2)}, {np.round(cy, 2)})")

for i in range(len(Hazards)):
    cx = Hazards[i][0]
    cy = Hazards[i][1]
    a = Hazards[i][2]
    b = Hazards[i][3]
    theta = np.linspace(0, 2 * np.pi, 100)  # Angle parameter for the ellipse
    x = a * np.cos(theta)  # Parametric x-coordinates
    y = b * np.sin(theta)  # Parametric y-coordinates
    z = np.zeros_like(x)  # Ellipse is on the xy-plane (z=0)

    # Translate the ellipse to its center (cx, cy)
    x_centered = x + cx
    y_centered = y + cy
    ax.plot(x_centered, y_centered, z, label=f"Hazard at ({cx}, {cy})")

ax.plot(predict[:, 0], predict[:, 1], predict[:, 2], label="Lyapunov Ctrl + UKF", color='m')
ax.plot(nominal[:, 0], nominal[:, 1], nominal[:, 2], label="Actual Trajectory", color='b', markersize = 0.01)
        

                
# Set labels
ax.legend(loc="best")
ax.set_xlabel('X axis (m)')
ax.set_ylabel('Y axis (m)')
ax.set_zlabel('Z axis (m)')
ax.set_title('Successive Convexification Optimal Trajectory')
plt.savefig("SCvx_Lyapunov_UKF.jpg", dpi = 300)
plt.show()