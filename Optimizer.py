from Parameters import *
from Linearize import *
from numpy import pi, ones
import cvxpy as cvx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from Dynamics import EOM
class Optimizer:
    
    def __init__(self, iter, IC, FC, GC, Hazards, Sigx):
        self.N = params['N']
        self.M = params['M']
        self.NUM = params['NUM']
        self.m0 = params['m0']
        self.h0 = params['h0']
        self.g = params['g']
        self.Isp = params['Isp']
        self.dt = params['dt']
        self.iter = iter
        self.IC = IC
        self.GC = GC
        self.Hazards = Hazards
        self.Sigx = Sigx
        self.Thrust_lim = params['Thrust_lim']
        self.alpha_lim = params['alpha_lim']
        self.beta_lim = params['beta_lim']
        self.fuel_lim = params['fuel_lim']
        self.const_vel = params['const_vel']
        self.tspan = params['t']
        self.trust_region = params['trust_region']
        self.delta_rate = params['delta_rate']
        self.const_vel_count = params['const_vel_count']
        self.inv_cdf = params['inv_cdf']
        self.Xiter = np.zeros((self.iter, self.N, self.NUM))
        self.Uiter = np.zeros((self.iter, self.M, self.NUM))

    def initParameters(self, Xnom, Unom):
        A, B = Linearize_guv(Xnom, Unom, [self.g, self.Isp])
        Ad, Bd, Cd = Discretize_guv(A, B, self.dt)
         
        Ad = np.transpose(Ad, (1, 2, 0))
        Bd = np.transpose(Bd, (1, 2, 0))
        Cd = np.transpose(Cd, (1, 2, 0))
        
        Ad = Ad.reshape(self.N * self.N, self.NUM)
        Bd = Bd.reshape(self.N * self.M, self.NUM)
        Cd = Cd.reshape(self.N * self.M, (self.NUM))
        
        self.Ah = cvx.Parameter((self.N * self.N, self.NUM - 1))
        self.Bh = cvx.Parameter((self.N * self.M, self.NUM - 1))
        self.Ch = cvx.Parameter((self.N * self.M, self.NUM - 1))
        self.Xprev = cvx.Parameter((self.N, self.NUM))
        self.Uprev = cvx.Parameter((self.M, self.NUM))
        
        self.Ah.value = Ad[:, :-1]
        self.Bh.value = Bd[:, :-1]
        self.Ch.value = Cd[:, :-1]
        
        self.Xprev.value = Xnom.T
        self.Uprev.value = Unom.T
        
        # Initialize Trust Region distance parameters
        self.delta = cvx.Parameter(nonneg = True)
        self.delta.value = self.trust_region
            
        return True
    
    def generateTrajectoryGuess(self):
        N = params['N']
        M = params['M']
        NUM = params['NUM']

        tspan = params['t']
        x0 = self.GC
        Xnom = odeint(EOM, x0, tspan, args=(params,), rtol=1e-13, atol=1e-13)
        Unom = np.column_stack([params['K'] * Xnom[:, -1] * params['g'], np.zeros(len(tspan)), np.zeros(len(tspan))])
        return Xnom, Unom

    def updateParameters(self, Xnom, Unom):
        A, B = Linearize_guv(Xnom, Unom, [self.g, self.Isp])
        Ad, Bd, Cd = Discretize_guv(A, B, self.dt)
         
        Ad = np.transpose(Ad, (1, 2, 0))
        Bd = np.transpose(Bd, (1, 2, 0))
        Cd = np.transpose(Cd, (1, 2, 0))
        
        Ad = Ad.reshape(self.N * self.N, self.NUM)
        Bd = Bd.reshape(self.N * self.M, self.NUM)
        Cd = Cd.reshape(self.N * self.M, (self.NUM))
        
        self.Ah.value = Ad[:, :-1]
        self.Bh.value = Bd[:, :-1]
        self.Ch.value = Cd[:, :-1]
        
        self.Xprev.value = Xnom.T
        self.Uprev.value = Unom.T
        
        self.delta.value = self.delta.value * self.delta_rate
            
        return True
        
    def initDecisionVariables(self):
        self.X = cvx.Variable((self.N, self.NUM)) # State Variables
        self.U = cvx.Variable((self.M, self.NUM)) # Control Input Variables
        self.NU = cvx.Variable((self.N, self.NUM - 1)) # Artificial Infeasibility
        self.NU_hazard = cvx.Variable((len(self.Hazards), 1)) # Artificial Infeasibility for Hazards
        return True
    
    def initConstraints(self, IC, FC):
        self.Constraints = []
        
        # Boundary Conditions
        self.Constraints += [self.X[0, 0] == IC[0], 
                             self.X[1, 0] == IC[1], 
                             self.X[2, 0] == IC[2],
                             self.X[3, 0] == IC[3], 
                             self.X[4, 0] == IC[4], 
                             self.X[5, 0] == IC[5],
                             self.X[6, 0] == IC[6],
                            self.X[2, -1] == 0,
                            self.X[6, -1] >= self.m0 - self.fuel_lim] 

        # Dynamics Constraints
        for i in range(0, self.NUM-1):
            self.Constraints += [self.X[:, i+1] == cvx.reshape(self.Ah[:, i], (self.N, self.N), order='C')@self.X[:, i]
                            +cvx.reshape(self.Bh[:, i], (self.N, self.M), order='C')@self.U[:, i]
                            +cvx.reshape(self.Ch[:, i], (self.N, self.M), order='C')@self.U[:, i+1]
                            + self.NU[:, i]] 
        
        # Control Constraints
        for i in range(0, self.NUM):
            self.Constraints += [self.U[0, i] >= self.Thrust_lim[0], 
                                 self.U[0, i] <= self.Thrust_lim[1], 
                                 self.U[1, i] >= self.alpha_lim[0] * (pi/180), 
                                 self.U[1, i] <= self.alpha_lim[1] * (pi/180),
                                 self.U[2, i] >= self.beta_lim[0] * (pi/180),
                                 self.U[2, i] <= self.beta_lim[1] * (pi/180)]
        # incremental Control Constraints
        for i in range(0, self.NUM-1):
            self.Constraints +=[cvx.abs(self.U[0, i] - self.U[0, i+1])<= 50, 
                                cvx.abs(self.U[1, i] - self.U[1, i+1])<= 3*(pi/180), 
                                cvx.abs(self.U[2, i] - self.U[2, i+1])<= 3*(pi/180)]
        
        for i in range(0, self.NUM-1):
            self.Constraints +=[cvx.abs(self.X[3, i] )<= 3, 
                                cvx.abs(self.X[4, i] )<= 3, 
                                cvx.abs(self.X[5, i] )<= 3]
        
        # Above x-y plane Constraint
        for i in range(0, self.NUM):
            self.Constraints += [self.X[0, i] >= 0, 
                                 self.X[1, i] >= 0, 
                                 self.X[2, i] >= 0]
        
        # Constant Velocity Descent Constraint
        for i in range(self.NUM - self.const_vel_count, self.NUM-1):
            self.Constraints += [self.X[0, i] == self.X[0, i+1], self.X[1, i] == self.X[1, i+1]]
            self.Constraints += [self.X[3, i] == 0, self.X[4, i] == 0, self.X[5, i] == self.const_vel]
        
        # Circle at origin hazard Constraint
        for i in range(len(self.Hazards)):
            c = np.array([[self.Hazards[i][0]], [self.Hazards[i][1]]])
            xb = np.array([[self.Xprev[0, -1].value], [self.Xprev[1, -1].value]])
            a = self.Hazards[i][2];
            b = self.Hazards[i][3];
            
            Q = np.array([[1/(a*a), 0], [0, 1/(b*b)]])
            bterm = (xb - c).T @ Q @ (xb - c) - 1
            aterm = 2*Q@(xb-c)
            var = np.sqrt(aterm.T @ self.Sigx @ aterm)
        
            # Regular Constraint
            #self.Constraints += [bterm + aterm[0, 0] * (self.X[0, -1] - xb[0, 0]) + aterm[1, 0] * (self.X[1, -1] - xb[1, 0]) + self.NU_hazard[i] >= 10**-12]
            # Chance Constraint
            self.Constraints += [bterm + aterm[0, 0] * (self.X[0, -1] - xb[0, 0]) + aterm[1, 0] * (self.X[1, -1] - xb[1, 0]) + self.NU_hazard[i] - self.inv_cdf * var >= 10**-12]
                                    
            
        # Trust Regions Constraint
        dX = self.X - self.Xprev
        dU = self.U - self.Uprev
        
        for i in range(1, self.NUM):
            self.Constraints += [cvx.norm(dX[:, i], 1) + cvx.norm(dU[:, i], 1) <= self.delta]
            
            
        return True
    def Optimize(self):
        
        Objective_Func = cvx.Minimize(10e4*cvx.norm(self.NU, 1) + cvx.sum(cvx.norm(self.U, 1, axis = 0)) + 10e4*cvx.norm(self.NU_hazard, 1))
        prob = cvx.Problem(objective= Objective_Func, constraints =  self.Constraints)    
        start = time.time()
        for i in range(self.iter):
            error = prob.solve(verbose=False, solver = cvx.ECOS, warm_start = True)
            self.updateParameters(self.X.value.T, self.U.value.T)
            print([i, np.sum(np.abs(self.NU.value))])
            self.Xiter[i, :, :] = self.X.value
            self.Uiter[i, :, :] = self.U.value
        print(time.time()-start)
        print(self.X.value[:, -1])
        print(self.X.value[:, 0])
        
        
    def PlotTrajectory(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(self.iter):
            if (i+1) % 4  == 0:
                ax.plot(self.Xiter[i, 0, :], self.Xiter[i, 1, :], self.Xiter[i, 2, :], label=f"Iter {i+1}")
        
        for i in range(len(self.Hazards)):
            cx = self.Hazards[i][0]
            cy = self.Hazards[i][1]
            a = self.Hazards[i][2]
            b = self.Hazards[i][3]
            theta = np.linspace(0, 2 * np.pi, 100)  # Angle parameter for the ellipse
            x = a * np.cos(theta)  # Parametric x-coordinates
            y = b * np.sin(theta)  # Parametric y-coordinates
            z = np.zeros_like(x)  # Ellipse is on the xy-plane (z=0)
            # Translate the ellipse to its center (cx, cy)
            x_centered = x + cx
            y_centered = y + cy
            ax.plot(x_centered, y_centered, z, label=f"Hazard at ({cx}, {cy})")
        
        cx = self.Xiter[-1, 0, -1]
        cy = self.Xiter[-1, 1, -1]
        a = np.sqrt(self.Sigx[0, 0])
        b = np.sqrt(self.Sigx[1, 1])
        theta = np.linspace(0, 2 * np.pi, 100)  # Angle parameter for the ellipse
        x = a * np.cos(theta)  # Parametric x-coordinates
        y = b * np.sin(theta)  # Parametric y-coordinates
        z = np.zeros_like(x)  # Ellipse is on the xy-plane (z=0)
        # Translate the ellipse to its center (cx, cy)
        x_centered = x + cx
        y_centered = y + cy
        ax.plot(x_centered, y_centered, z, label=f"99% Confidence Zone ({np.round(cx, 2)}, {np.round(cy, 2)})")
                
        # Set labels
        ax.legend(loc="best")
        ax.set_xlabel('X axis (m)')
        ax.set_ylabel('Y axis (m)')
        ax.set_zlabel('Z axis (m)')
        ax.set_title('Successive Convexification Optimal Trajectory')
        plt.show()
    
        return True
    def PlotStates(self):
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle("Trajectory", fontsize=16)
        ax = axes[0, 0]
        ax.plot(self.tspan, self.Xiter[-1, 0, :])
        ax.set_ylabel("X (m)")
        ax.set_xlabel("time (sec)")
        
        ax = axes[0, 1]
        ax.plot(self.tspan, self.Xiter[-1, 1, :])
        ax.set_ylabel("Y (m)")
        ax.set_xlabel("time (sec)")

        ax = axes[0, 2]
        ax.plot(self.tspan, self.Xiter[-1, 2, :])
        ax.set_ylabel("Z (m)")
        ax.set_xlabel("time (sec)")

        ax = axes[1, 0]
        ax.plot(self.tspan, self.Xiter[-1, 3, :])
        ax.set_ylabel("Vx (m/s)")
        ax.set_xlabel("time (sec)")


        ax = axes[1, 1]
        ax.plot(self.tspan, self.Xiter[-1, 4, :])
        ax.set_ylabel("Vy (m/s)")
        ax.set_xlabel("time (sec)")


        ax = axes[1, 2]
        ax.plot(self.tspan, self.Xiter[-1, 5, :])
        ax.set_ylabel("Vz (m/s)")
        ax.set_xlabel("time (sec)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        
        ax = axes[0, 3]
        ax.plot(self.tspan, self.Xiter[-1, 6, :])
        ax.set_ylabel("Fuel Mass (kg)")
        ax.set_xlabel("time (sec)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        ax = axes[0, 4]
        ax.plot(self.tspan, self.Uiter[-1, 0, :])
        ax.set_ylabel("Thrust (N)")
        ax.set_xlabel("time (sec)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        ax = axes[1, 3]
        ax.plot(self.tspan, self.Uiter[-1, 1, :])
        ax.set_ylabel("alpha (rad)")
        ax.set_xlabel("time (sec)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        ax = axes[1, 4]
        ax.plot(self.tspan, self.Uiter[-1, 2, :])
        ax.set_ylabel("beta (rad)")
        ax.set_xlabel("time (sec)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return True
    def test(self):
        self.Xtest = np.zeros((self.N, self.NUM))
        self.Xtest[:, 0] = self.X.value[:, 0]
        for i in range(self.NUM-1):
            self.Xtest[:, i+1] = cvx.reshape(self.Ah[:, i], (self.N, self.N), order='C').value@self.Xtest[:, i]
            +cvx.reshape(self.Bh[:, i], (self.N, self.M), order='C').value@self.U[:, i].value
            +cvx.reshape(self.Ch[:, i], (self.N, self.M), order='C').value@self.U[:, i+1].value
        self.X.value = self.Xtest
                            
        
    
'''    
IC = [0, 0, params['h0'], 0, 0, 0, params['m0']]
FC = [15.00, 20.00, 0, 0, 0, 0, params['m0']-500]
GC = [40, 40, params['h0'], 0, 0, 0, params['m0']]
Hazards = [[0, 0, 20, 20], [30, 30, 20, 20]]
Sigx = np.array([[200, 0], [0, 500]])
opt = Optimizer(50, IC, FC, GC, Hazards, Sigx)
Xnom, Unom = opt.generateTrajectoryGuess()
opt.initParameters(Xnom, Unom)
opt.initDecisionVariables()
opt.initConstraints(IC, FC)
opt.Optimize()
#opt.test()
opt.PlotTrajectory()

np.savez('Nominal_Trajectory.npz', Xnom = opt.X.value, Unom = opt.U.value)
'''