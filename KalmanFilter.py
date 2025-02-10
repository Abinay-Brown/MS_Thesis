import numpy as np
from Dynamics import EOM, dynamics
from numpy import pi, sin, cos
from Parameters import *
from Integrators import rk4_step, rk8_step

def Sensor_Measurement(ind, state_int, statedot, state_next):
    dt = 1/params['control_freq']
    noise = np.sqrt(ukf_params['acc_sig_sq']) * np.array([np.random.normal(0, 1), np.random.normal(0, 1), np.random.normal(0, 1)])
    accelerometer = statedot[3:-1] + noise
    v_int = state_int[3:] + accelerometer * dt
    p_int = state_int[0:3]  + v_int * dt
    state_int = np.hstack((p_int, v_int))
    altimeter = state_next[2] + np.sqrt(ukf_params['alt_sig_sq']) * np.random.normal(0, 1)
    z = np.hstack((state_int, altimeter)).T
    # Update R matrix
    R = np.diag([((ind+1)**2)*ukf_params['acc_sig_sq']*dt**4, ((ind+1)**2)*ukf_params['acc_sig_sq']*dt**4, ((ind+1)**2)*ukf_params['acc_sig_sq']*dt**4, 
             (ind+1)*ukf_params['acc_sig_sq']*dt**2, (ind+1)*ukf_params['acc_sig_sq']*dt**2, (ind+1)*ukf_params['acc_sig_sq']*dt**2, ukf_params['alt_sig_sq']])
    
    return z, R, state_int
def UKF_update(t, state_est, Pukf, z, R, ctrl):
    Q = ukf_params['Q']
    W = ukf_params['W']
    
    dt = 1/params['control_freq']
    # Generate Sigma Points
    root = np.linalg.cholesky(params['N']*Pukf)
    xbreve = np.zeros((params['N'], 2*params['N'] +1))
    xbreve[:, 0] = state_est
    for i in range(params['N']):  
        xbreve[:, i + 1] = state_est + root[:, i]  
        xbreve[:, i + params['N'] + 1] = state_est - root[:, i] 
    
    # Ensure mass remains unchanged in all sigma points
    #xbreve[-1, 1:] = state_est[-1]  # Keep mass constant

    # Propagate sigma points
    for i in range(2*params['N'] +1):
        state_new  = rk4_step(dynamics, t, xbreve[:, i], dt/100, ctrl)
        xbreve[:, i] = state_new
    
    # Average the Propagated sigma points
    state_est = np.zeros((params['N']))
    for i in range(2*params['N'] + 1):
        state_est = state_est + xbreve[:, i]
        
    state_est /= W
    
    # Estimate the Covariance Matrix
    Pukf = np.zeros((params['N'], params['N']))
    for i in range(2*params['N'] + 1):   
        diff = xbreve[:, i] - state_est
        Pukf = Pukf + np.outer(diff,  diff)
    
    Pukf /= W
    Pukf = Pukf + Q
    
    # UKF Measurement Update
    zukf = np.zeros((len(z), 2 * params['N'] + 1))  
    zhat = np.zeros(len(z)) 

    for i in range(2 * params['N'] + 1):
        zukf[:, i] = np.array([xbreve[0, i], xbreve[1, i], xbreve[2, i], xbreve[3, i], xbreve[4, i], xbreve[5, i], xbreve[2, i]])
        zhat += zukf[:, i] / W
    
    
    Py = np.zeros((len(z), len(z))) 
    Pxy = np.zeros((params['N'], len(z))) 

    for i in range(2 * params['N'] + 1):
        diff_z = zukf[:, i] - zhat  
        diff_x = xbreve[:, i] - state_est  
        Py += np.outer(diff_z, diff_z) / W 
        Pxy += np.outer(diff_x, diff_z) / W

    Py += R

    Kukf = Pxy @ np.linalg.inv(Py)  

    # Update state estimate
    state_est = state_est + Kukf @ (z - zhat)    
    
    
    return state_est, Pukf