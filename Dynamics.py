import numpy as np
from numpy import sin, cos, eye, dot, linspace, pi
from Parameters import *

def EOM(state, t, params):
    g = params['g']
    Isp = params['Isp']
    K = params['K']
    
    x, y, z = state[0], state[1], state[2]
    xd, yd, zd = state[3], state[4], state[5]
    m = state[6]
    
    T = K * m * g
    alpha = 0
    beta = 0
    
    ax = -(T / m) * cos(alpha) * sin(beta)
    ay = (T / m) * sin(alpha)
    az = (T / m) * cos(alpha) * cos(beta) - g
    
    mdot = -T / (g * Isp)

    statedot = [xd, yd, zd, ax, ay, az, mdot]
    return statedot

def dynamics(t, state, control):
    g = params['g']
    Isp = params['Isp']
    K = params['K']
    
    x, y, z = state[0], state[1], state[2]
    xd, yd, zd = state[3], state[4], state[5]
    m = state[6]
    
    T = control[0]
    alpha = control[1]
    beta = control[2]
    
    ax = -(T / m) * cos(alpha) * sin(beta)
    ay = (T / m) * sin(alpha)
    az = (T / m) * cos(alpha) * cos(beta) - g
    
    mdot = -T / (g * Isp)
    
    statedot = np.array([xd, yd, zd, ax, ay, az, mdot])
    return statedot

