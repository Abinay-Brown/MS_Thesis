A = [0, 0, 0, 1, 0, 0, 0;
0, 0, 0, 0, 1, 0, 0.;
0, 0, 0, 0, 0, 1, 0.;
0, 0, 0, 0, 0, 0, 0.;
0,  0,  0,  0,  0,  0, -0.;
0, 0, 0, 0, 0, 0, -0.00100923;
0, 0, 0, 0, 0, 0, 0]
dt = 0.6060606060606061
Ad = expm(A*dt)
format longg
B1d = zeros(7, 3)
B2d = zeros(7, 3)
N = 500;
B = [0, 0, 0;
      0, 0, 0;
      0, 0, 0;
      0, 0, -1.51385;
      0, 1.51385, 0;
      0.00066667, 0, 6;
      -0.00175824, 0, 0];

for step = 1:N
    dtau = dt/N;
    tau = step * dtau;
    Texp = dt - tau;
    B1d = B1d + expm(A*(dt-tau))*B*dtau;
    B2d = B2d + expm(A*(dt-tau))*B*(tau/dt)*dtau;
end
B1d
B2d