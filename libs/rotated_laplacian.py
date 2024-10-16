from .pde import *
from .utils import *
import time
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy as sp
import pyamg
from torch.autograd import Variable
from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
from scipy.sparse import csr_matrix, isspmatrix_csr

class RotatedLaplacian(PDEData):
    def __init__(self,
                 grid_size : int,
                 k : list,
                 epsilonList : list,
                 thetaList : list,
                 dataset_option : str,
                 debug = False,
                 same_vecs = False,
                 ):
        """
        grid_size : size of training mesh
        k : list of integers
            k[0] contains the number of smooth vectors in level 2
            k[1] contains the number of smooth vectors in level 3
        dataset_option available : {'fixed xi', 'fixed theta'}
        same_vecs : if true, smooth vectors on level 3 will be the same as those on level 2
        
        Data generation has only been implemented for <= 3-level training

        
        """
        super().__init__()
        self.n = grid_size
        self.epsList = epsilonList
        self.thetaList = thetaList
        self.k = k
        self.dataset_option = dataset_option

        if debug:
            self.epsList = [1.0]
            self.thetaList = [np.pi/4]
            self.k = [10,10]
        self.A2_train, self.A3_train, self.s2_train, self.s3_train, self.eig_vec2_train, self.eig_vec3_train = \
        self.generate_data(self.n, self.k, self.epsList, self.thetaList, self.dataset_option)
        """
        type of the variables in line 69 : list
        """

        if same_vecs:
            self.eig_vec3_train = self.eig_vec2_train

    def generate_data(self, n : int, k : list, epsList, thetaList, dataset_option : str):
        """
        epsList : list or ndarray
        thetaList : list or ndarray
        dataset_option : string
        """
        # type casting
        if type(epsList) is not list:
            epsList = list(epsList)
        if type(thetaList) is not list:
            thetaList = list(thetaList)
        
        # shape check
        if dataset_option == 'fixed xi':
            assert len(epsList) == 1
            npts = len(thetaList)
            epsList = epsList * npts
        elif dataset_option == 'fixed theta':
            assert len(thetaList) == 1
            npts = len(epsList)
            thetaList = thetaList * npts
        else:
            raise KeyError("Dataset Option is not available")
        
        params = zip(epsList, thetaList)
        
        A2_train = []
        A3_train = []
        s2_train = []
        s3_train = []
        eig_vec2_train = []
        eig_vec3_train = []

        for param in params:
            eps, theta = param
            A2_conv, A3_conv, s2, s3, eig_vec2, eig_vec3 = self.gen_data_point(n, k, eps, theta)
            A2_train.append(A2_conv)
            A3_train.append(A3_conv)
            s2_train.append(torch.from_numpy(s2))
            s3_train.append(torch.from_numpy(s3))
            eig_vec2_train.append(eig_vec2)
            eig_vec3_train.append(eig_vec3)
        
        return A2_train, A3_train, s2_train, s3_train, eig_vec2_train, eig_vec3_train
    
    def train(self,
              A_train, 
              s_train,
              eig_vec_train,
              model_prob,
              model_value,
              epochs,
              adam_decay_rate,
              lr,
              lr_decay_rate,
              lr_decay_step,
              device,
              enforce_stencil_symmetry : str = None,
              single_model = False,
              softmax_on = True,
              verbose = False):

        model_prob = model_prob.to(device).double()
        model_value = model_value.to(device).double()
        optimizer = torch.optim.Adam(list(model_prob.parameters())+list(model_value.parameters()), 
                                    lr=lr, 
                                    weight_decay=adam_decay_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=lr_decay_step,
                                                    gamma=lr_decay_rate)
        if enforce_stencil_symmetry is not None:
            if enforce_stencil_symmetry not in ['+','x','+2','x2']:
                raise ValueError(f"{enforce_stencil_symmetry} type stencil symmetry is not implemented!")

        for epoch in range(epochs):
            loss = 0
            optimizer.zero_grad()
            for j in range(len(A_train)):
                A = A_train[j].to(device).double()
                stencil0 = s_train[j]
                eig_vec = eig_vec_train[j]
                eig_vec = Variable(eig_vec.clone()).to(device).double()
                stencil = Variable(stencil0.clone().squeeze(0).squeeze(0)).reshape(9,1)
                stencil = torch.cat([stencil[0:4],stencil[5:9]]).t().to(device).double()
                prob = model_prob(stencil).squeeze()
                value = model_value(stencil).squeeze()
#                 print(value)
#                 print(value.size(0))
#                 assert value.size(0) == 1

                # sparsify stencil
                coarse_stencil= self.sparsify(prob,value,single_model,softmax_on,enforce_stencil_symmetry, device)
                
                # compute loss
                temp = F.conv2d(eig_vec,coarse_stencil,padding=1)-A(eig_vec)
                temp = temp.squeeze(1).view(temp.shape[0],-1,1)
                loss += torch.norm(temp) ** 2
            loss.backward()
            optimizer.step()
            scheduler.step()
            # for name, param in model_value.named_parameters():
            #     print(name, param.grad)
            if verbose:
                if epoch % 250 == 0 or epoch==(epochs-1):
                    print(' epoch: ',epoch,' loss: ',loss)
        return model_prob,model_value


    def test_model(self,
                   num_test : int,
                   models : dict,
                   test_grid_size : int,
                   max_levels : int,
                   epsilon : tuple,
                   theta : tuple,
                   device,
                   dropout_on = False,
                   enforce_stencil_symmetry : str = None,
                   random_test : bool = False,
                   single_model = False,
                   softmax_on = True,
                   top_accel = None,
                   verbose = True):
        
        """
        theta & epsilon : parameter tuple
                          can only be of length 2 or 1

                          if testing on epsilon interval, then 
                          theta = (theta), epsilon = (eps_low, eps_high)

                          if testing on theta interval, then 
                          theta = (theta_low, theta_high), epsilon = (epsilon)
        
        """

        if enforce_stencil_symmetry is not None:
            if enforce_stencil_symmetry not in ['+','x','+2','x2']:
                raise ValueError(f"{enforce_stencil_symmetry} type stencil symmetry is not implemented!")

        # type casting
        if type(epsilon) is not tuple:
            raise ValueError("Epsilon should be of type tuple. If it is a single number, input epsilon=(number)")
        if type(theta) is not tuple:
            raise ValueError("Theta should be of type tuple. If it is a single number, input theta=(number)")
        
        # shape check
        if len(epsilon)==1:
            assert len(theta) == 2
            eps = [epsilon[0]] # cast into a list 
            theta_low, theta_high = theta
            epsList = eps * num_test
            assert type(epsList) is list
            if random_test:
                thetaList = [np.random.uniform(low=theta_low, high=theta_high) for _ in range(num_test)]
            else:
                h = (theta_high - theta_low)/(num_test + 1) # exclude end points in test cases
                thetaList = [theta_low + h*(i+1) for i in range(num_test)]
                assert theta_low not in thetaList and theta_high not in thetaList

        else:
            assert len(epsilon) == 2
            assert len(theta) == 1
            theta = [theta[0]]
            eps_low,  eps_high = epsilon
            thetaList = theta * num_test
            assert type(thetaList) is list
            if random_test:
                epsList = [np.random.uniform(low=eps_low, high=eps_high) for _ in range(num_test)]
            else:
                h = (eps_high - eps_low)/(num_test + 1) # exclude end points in test cases
                epsList = [eps_low + h*(i+1) for i in range(num_test)]
                assert eps_low not in epsList and eps_high not in epsList

        paramList = zip(epsList, thetaList)
        
        res_standard = []
        res_learning = []
        num_iter_standard = []
        num_iter_learning = []
        t_standard = []
        t_learning = []

        for param in paramList:
            eps, theta = param
            s = np.double(self.get_stencil(eps, theta))
            A = pyamg.gallery.stencil_grid(s,(test_grid_size,test_grid_size)).tocsr()

            solver_standard, A2, A3 = self.geometric_solver(A,
                                                         'standard',
                                                         models,
                                                         test_grid_size,
                                                         dropout_on = dropout_on,
                                                         max_levels = max_levels,
                                                         single_model = single_model,
                                                         enforce_stencil_symmetry = enforce_stencil_symmetry,
                                                         softmax_on = softmax_on,
                                                         coarse_solver='splu')
            # assert 0 == 1 
            solver_non_galerkin, A2s, A3s= self.geometric_solver(A,
                                                        'learning',
                                                        models,
                                                        test_grid_size,
                                                        dropout_on=dropout_on,
                                                        max_levels=max_levels,
                                                        coarse_solver='splu',
                                                        single_model = single_model,
                                                        enforce_stencil_symmetry=enforce_stencil_symmetry,
                                                        softmax_on=softmax_on,
                                                        device=device)

            x0 = np.ones((A.shape[0],1))
            b = np.random.rand(A.shape[0],1)

            t1 = time.time()
            xs = solver_standard.solve(b,x0=x0,maxiter=1000, tol=1e-6,residuals=res_standard, accel=top_accel)
            t2 = time.time()
            xl = solver_non_galerkin.solve(b,x0=x0,maxiter=1000, tol=1e-6,residuals=res_learning, accel=top_accel)
            t3 = time.time()

            num_iter_standard.append(len(res_standard))
            num_iter_learning.append(len(res_learning))
#             if num_iter_learning[-1] > 2*num_iter_standard[-1]:
#                 break

            t_standard.append(t2-t1)
            t_learning.append(t3-t2)

        standard_iter_avg = np.mean(num_iter_standard)
        learning_iter_avg = np.mean(num_iter_learning)
        if verbose:
            print('standard stencil iter:   ',standard_iter_avg, \
                '  standard stencil time:  ', np.mean(t_standard))

            print('learned stencil iter:   ',learning_iter_avg, \
                '  learned stencil time:  ',np.mean(t_learning))
        
        return num_iter_standard, num_iter_learning, epsList, thetaList, xs, xl, A2, A3, A2s, A3s

    def gen_data_point(self, n : int, k : list, epsilon, theta):
        s = self.get_stencil(epsilon, theta)
        k1 = k[0]
        k2 = k[1]
        A = pyamg.gallery.stencil_grid(s,(n,n)).tocsr()
        R = self.get_prolongation(n)
        P = R.T*4
        T = R @ P
        A = R @ A @ P

        _, eig_vec = sp.sparse.linalg.eigs(A,k=k1,M=T,which = 'SM')
        eig_vec = eig_vec.T
        eig_vec = torch.real(torch.from_numpy(eig_vec)).view(k1,1,n//2,n//2)

        # level 1 data
        stencils = compute_stencil(A,n//2,n//2,3)
        s1 = stencils[n//4,n//4,:,:]
        AA_test = pyamg.gallery.stencil_grid(s1,(n//2,n//2))
        assert np.linalg.norm(A.toarray()-AA_test.toarray())==0.0
        A_conv = nn.Conv2d(1, 1, 3, padding = 1, bias = False).double()
        A_conv.weight = nn.Parameter(torch.from_numpy(s1).view(1,1,3,3),requires_grad = False)

        # level 2 data
        R2 = self.get_prolongation(n//2)
        P2 = R2.T*4
        A2 = R2@A@P2
        T2 = R2 @ P2
        _, eig_vec2 = sp.sparse.linalg.eigs(A2,k=k2,M=T2,which = 'SM')
        eig_vec2 = eig_vec2.T
        eig_vec2 = torch.real(torch.from_numpy(eig_vec2)).view(k2,1,n//4,n//4)

        stencils2 = compute_stencil(A2,n//4,n//4,3)
        s2 = stencils2[n//8,n//8,:,:]
        A2_test = pyamg.gallery.stencil_grid(s2,(n//4,n//4))
        assert np.linalg.norm(A2.toarray()-A2_test.toarray()) == 0.0
        A2_conv = nn.Conv2d(1, 1, 3, padding = 1, bias = False).double()
        A2_conv.weight = nn.Parameter(torch.from_numpy(s2).view(1,1,3,3),requires_grad = False)
        # with np.printoptions(threshold=np.inf):
        #     print('A2:',sp.sparse.csr_matrix.todense(A2))
        #     print('A2_test:',sp.sparse.csr_matrix.todense(A2_test))
        #     print('s2:', s2)

        return A_conv, A2_conv, s1, s2, eig_vec, eig_vec2
    
    def get_stencil(self, eps, theta):
        """
        generate the stencil for 2-D Laplacian equation
        """
        return diffusion_stencil_2d(epsilon=eps, theta=theta, type='FD')

    def get_prolongation(self, grid_size):
        """
        prolongation operator
        """
        res_stencil = np.double(np.zeros((3,3)))
        d=16
        res_stencil[0,0] = 1/d
        res_stencil[0,1] = 2/d
        res_stencil[0,2] = 1/d
        res_stencil[1,0] = 2/d
        res_stencil[1,1] = 4/d
        res_stencil[1,2] = 2/d
        res_stencil[2,0] = 1/d
        res_stencil[2,1] = 2/d
        res_stencil[2,2] = 1/d
        P_stencils= np.zeros((grid_size//2,grid_size//2,3,3))
        for i in range(grid_size//2):
            for j in range(grid_size//2):
                P_stencils[i,j,:,:]=res_stencil
        return compute_p2_old(P_stencils, grid_size).astype(np.double)

    def geometric_solver(self, A,
                         option1,  
                         models,
                         n, # grid size
                         presmoother=('gauss_seidel', {'sweep': 'forward'}),
                         postsmoother=('gauss_seidel', {'sweep': 'forward'}),
                         max_levels=3, 
                         max_coarse=10,
                         dropout_on=False,
                         coarse_solver='splu',
                         device=None,
                         single_model = False,
                         enforce_stencil_symmetry = None,
                         softmax_on=True,
                         **kwargs):
        """
        models - dictionary
                 models['level2'] contains trained models for sparsification on level 3
                 models['level3'] contains trained models for sparsification on level 3
        """
    
        levels = [multilevel_solver.level()]
        

        # convert A to csr
        if not isspmatrix_csr(A):
            try:
                A = csr_matrix(A)
            except BaseException:
                raise TypeError('Argument A must have type csr_matrix, \
                                or be convertible to csr_matrix')
        # preprocess A
        A = A.asfptype()
        if A.shape[0] != A.shape[1]:
            raise ValueError('expected square matrix')

        levels[-1].A = A
        levels[-1].Ag = A
        levels[-1].n = n

        while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
            self.extend_hierarchy(levels, option1, models, device, single_model, softmax_on, 
                                  enforce_stencil_symmetry, dropout_on=dropout_on)
        if max_levels == 2:
            A3 = None
        else:
            A3 = levels[-1].A
        A2 = levels[1].A
            
        ml = multilevel_solver(levels, coarse_solver=coarse_solver)
        
        change_smoothers(ml, presmoother, postsmoother)
        return ml, A2, A3

    def extend_hierarchy(self, levels, option1, models, device, single_model, softmax_on, 
                         enforce_stencil_symmetry, dropout_on=False):
        """Extend the multigrid hierarchy."""
        A = copy.deepcopy(levels[-1].Ag)
        n = levels[-1].n
        R = self.get_prolongation(n)
        P = R.T*4
        levels[-1].P = P  # prolongation operator
        levels[-1].R = R  # restriction operator
        levels.append(multilevel_solver.level()) # add a new level
        n = n//2
        levels[-1].n = n

        if option1=='standard':
            A_g = R@A@P
            A_g = A_g.astype(np.float64)  # convert from complex numbers, should have A.imag==0
            levels[-1].A = A_g.tocsr()
            levels[-1].Ag = levels[-1].A
            stencils = compute_stencil(A_g, n, n, 3)
            s = stencils[n//2,n//2,:,:].squeeze()
#             print(s)
#             print(A_g == A_g.T)
            # print('standard: ', A.count_nonzero() / A.shape[1])

        elif option1=='learning':
            # if len(levels) == 1:
            level_num = len(levels)
#             if level_num != 3:
            model_key = f"level{level_num}"
            model_prob, model_value = models[model_key]
            if not dropout_on:
                model_prob.eval()
                model_value.eval()
            ### TODO?: use sparsified level 2 stencil to train the level 3 model

            A_g = R@A@P
            levels[-1].Ag = A_g
            if n < 3:
                raise ValueError('Stencil at this level is not available because the size of the \
                                training grid is too small! Need to increase the \'n\'!')
            
            stencils = compute_stencil(A_g, n, n, 3)
            s = stencils[n//2,n//2,:,:].squeeze()
#             print(s)
#             print(A_g == A_g.T)
            stencil = s.reshape(9,1)
            stencil = torch.from_numpy(stencil)
            stencil = torch.cat([stencil[0:4],stencil[5:9]]).t()
#             if level_num != 3:
            with torch.no_grad():
                prob = model_prob(stencil.to(device).double()).squeeze()
                value = model_value(stencil.to(device).double()).squeeze()
            XX = self.sparsify(prob,value,single_model,softmax_on,enforce_stencil_symmetry,device)
#             print('XX:', XX)
            A_c = pyamg.gallery.stencil_grid(XX.squeeze().detach().cpu().numpy(),(n,n))
            levels[-1].A = A_c
#             else:
#                 levels[-1].A = A_g
            # print('learning: ',A_c.count_nonzero()/A_c.shape[1])

        else:
            raise ValueError('Option1 is not available!')

    def sparsify(self, prob, value, single_model, softmax_on, enforce_symmetry, device):
        """
        sparsify stencil

        """
        if single_model:
            # choose top values instead of top absolute values:
            idx = torch.topk(value,4)[1]
            val = torch.topk(value,4)[0]
            # stencil = ...
            # ...
            # choose top absolute values:
            # idx = torch.topk(torch.abs(value),4)[1]
            # sign_val = torch.sign(value)[idx]
            # val = torch.topk(torch.abs(value),4)[0]
            # val = val * sign_val
            stencil = torch.zeros_like(value,memory_format=torch.legacy_contiguous_format)
            stencil = stencil.scatter_(-1,idx,val)
            stencil = stencil - value.detach() + value
            stencil = stencil.squeeze()
            return torch.cat((stencil[0:4],-stencil.sum().view(1),stencil[4:8])).view(1,1,3,3)
        else:
            if enforce_symmetry is None:
                stencil = top_k(prob,4,softmax_on).squeeze()
                stencil = stencil * value
                stencil = torch.cat((stencil[0:4],-stencil.sum().view(1),stencil[4:8])).view(1,1,3,3)
            else:
                # enforce value symmetry
                value = value + torch.flip(value,[0])
                if enforce_symmetry == '+':
                    # enforce position symmetry
                    prob[0] = 0
                    prob[1] = 1
                    prob[2] = 0
                    prob[3] = 1
                    prob[4] = 1
                    prob[-1] = 0
                    prob[-2] = 1
                    prob[-3] = 0
                    stencil = prob * value
                    stencil = torch.cat((stencil[0:4],-stencil.sum().view(1),stencil[4:8])).view(1,1,3,3)
#                     print(stencil)
                elif enforce_symmetry == 'x':
                    prob[0] = 1
                    prob[1] = 0
                    prob[2] = 1
                    prob[3] = 0
                    prob[4] = 0
                    prob[-1] = 1
                    prob[-2] = 0
                    prob[-3] = 1
                    stencil = prob * value
                    stencil = torch.cat((stencil[0:4],-stencil.sum().view(1),stencil[4:8])).view(1,1,3,3)
                elif enforce_symmetry == '+2':
                    value = value.reshape(2,1)
                    assert value.shape == (2,1)
                    matrix = torch.tensor([[0,0],
                                           [1,0],
                                           [0,0],
                                           [0,1],
                                           [0,1],
                                           [0,0],
                                           [1,0],
                                           [0,0]])
                    matrix = matrix.to(device)
                    stencil = matrix @ value
                    stencil = torch.cat((stencil[0:4],-stencil.sum().view(1),stencil[4:8])).view(1,1,3,3)
                elif enforce_symmetry == 'x2':
                    value = value.reshape(2,1)
                    assert value.shape == (2,1)
                    matrix = torch.tensor([[1,0],
                                           [0,0],
                                           [0,1],
                                           [0,0],
                                           [0,0],
                                           [0,1],
                                           [0,0],
                                           [1,0]])
                    matrix = matrix.to(device)
                    stencil = matrix @ value
                    stencil = torch.cat((stencil[0:4],-stencil.sum().view(1),stencil[4:8])).view(1,1,3,3)
            return stencil
