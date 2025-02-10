import numpy as np

# Define parameters
params = {
    'N': 7,
    'M': 3,
    'g': 1.625,
    'Isp': 350,
    'sigma': 60,
    'NUM': 50,
    'm0': 1500,
    'h0': 200,
    'K': 0.9316,
    'T': 60,
    'Thrust_lim': [1, 1500*1.625],
    'alpha_lim': [-30, 30],
    'beta_lim': [-30, 30],
    'fuel_lim': 500,
    'trust_region': 500,
    'delta_rate': 0.92,
    'const_vel': -2,
    'const_vel_count': 15,
    'inv_cdf': 2.33, # 99%: 2.33, 95%: 1.645, 90%: 1.28
    'control_freq': 250
}

# Create the linspace and calculate additional variables
params['tau'] = np.linspace(0, 1, params['NUM'])
params['dtau'] = params['tau'][1] - params['tau'][0]
params['t'] = params['tau'] * params['sigma']
params['dt'] = params['tau'][1] * params['T']
#print(params['tau']);

ukf_params = {
    'freq': 250,
    'dt': 1/250,
    'acc_sig_sq': 200*((35*10**-6)*params['g'])**2,
    'alt_sig_sq': 0.2,
    'Pukf': np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 1e-20]),
    'Q': np.diag([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 1e-20]),
    'W': 2 * params['N'] + 1
}