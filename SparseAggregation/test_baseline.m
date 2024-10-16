addpath('MEXfunc');

%% 

disp("############### START ##############")
theta = 4;
xi_low = 0.01;
xi_hi = 0.0125;

theta = num2str(theta);
xi_low = num2str(xi_low);
xi_hi = num2str(xi_hi);
exp_dir = strcat('data/matrix/theta-',theta,'/xi-',xi_low,'-',xi_hi,'/');
disp(exp_dir);

matfiles = dir(strcat(exp_dir,'*.mat')) ; 
numfiles = 10;
disp(numfiles)
disp(matfiles)
num_its = [];
mat_nnz = [];
nnz_rate = [];
for i = 1:numfiles
    load(strcat(exp_dir,matfiles(i).name))
    no = length(A);
    mat_nnz(i) = nnz(A);
    bo = rand(no,1);
    AT = A';
    [xo,ao_setup_info,ao_res] = SolveLinearSystem(AT,bo);
    num_its(i) = ao_res{1}.iter;
%     nnz_rate(i) = l2mean_nnz;
    disp("relative error of A*xo=bo:")
    norm(A*xo-bo)/norm(bo)
end
disp(strcat('Average #Iterations:', num2str(mean(num_its,'all'))))
disp(num_its)
disp('Level 1 NNZ: ')
disp(mat_nnz)
% disp('Level 2 NNZ: ')
% disp(nnz_rate)
disp("############### DONE ##############")


%%
% ng = length(Ag);
% bg = rand(ng,1);
% % disp(n)
% [xg,ag_setup_info,ag_res] = SolveLinearSystem(Ag',bg);
% disp("relative error of Ag*xg=bg:")
% norm(Ag*xg-bg)/norm(bg)

%%
% nc = length(Ac);
% bc = rand(nc,1);
% [xc,ac_setup_info,ac_res] = SolveLinearSystem(Ac',bc);
% disp("relative error of Ac*xc=bc:")
% norm(Ac*xc-bc)/norm(bc)
