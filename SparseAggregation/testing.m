
addpath('MEXfunc');
loader = load("/MATLAB Drive/Sparsification/data/xi-100-200_theta-pi_2_0.mat"); % this line saves the matrix as A
A = getfield(loader,'A');
% A1 = getfield(load('Sparsification/data/matrix/'), 'A'); % this line saves the matrix as a new variable

%% 
n = length(A);
b = rand(n,1);
[xo,ao_setup_info,ao_res] = SolveLinearSystem(A,b);