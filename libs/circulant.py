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
from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
from scipy.sparse import csr_matrix, isspmatrix_csr
from pyamg.gallery import stencil_grid

class Circulant(PDEData):
    def __init__(self,train_grid_size : int, k : list):
        super().__init__()
        self.n = train_grid_size
        self.k = k
        self.s, self.A2, self.A3, self.s2, self.s3, self.eig_vec2, self.eig_vec3 = \
                                                self.generate_data(self.k, self.n)
        self.optimal_s2 = self.optimal_stencil(self.s2)
        self.optimal_s3 = self.optimal_stencil(self.s3)

    def test_model(self,
                   models : dict,
                   test_grid_size : int,
                   max_levels : int,
                   device,
                   single_model = False,
                   softmax_on = True,
                   top_accel = None,
                   enforce_pos = False,
                   verbose = True):
        """
        models: models[f"level{k}"] gives the sparsification models on the k-th level

        """
        A = stencil_grid(self.s,(test_grid_size,test_grid_size))
        solver_standard = self.geometric_solver(A,
                                                'standard',
                                                models,
                                                test_grid_size,
                                                max_levels=max_levels,
                                                single_model = single_model,
                                                softmax_on=softmax_on,
                                                enforce_pos=enforce_pos,
                                                coarse_solver='splu')
        # assert 0 == 1 
        solver_learn = self.geometric_solver(A,
                                             'learning',
                                             models,
                                             test_grid_size,
                                             max_levels=max_levels,
                                             coarse_solver='splu',
                                             single_model = single_model,
                                             softmax_on=softmax_on,
                                             enforce_pos=enforce_pos,
                                             device=device)
        x0 = np.ones((A.shape[0],1))
        b = np.random.rand(A.shape[0],1)
        sl = []
        sl.append(solver_learn.levels[1].s)
        if max_levels == 3:
            sl.append(solver_learn.levels[-1].s)

        res_standard = []
        res_learning = []
        t1 = time.time()
        xs = solver_standard.solve(b,x0=x0,maxiter=1000, tol=1e-6,residuals=res_standard, accel=top_accel)
        t2 = time.time()
        xl = solver_learn.solve(b,x0=x0,maxiter=1000, tol=1e-6,residuals=res_learning, accel=top_accel)
        t3 = time.time()

        num_iter_standard = len(res_standard)
        num_iter_learning = len(res_learning)
#             if num_iter_learning[-1] > 2*num_iter_standard[-1]:
#                 break

        t_standard = t2 - t1
        t_learning = t3 - t2

        if verbose:
            print('standard stencil iter:   ',num_iter_standard, \
                '  standard stencil time:  ', t_standard)

            print('learned stencil iter:   ',num_iter_learning, \
                '  learned stencil time:  ',t_learning)
        
        return sl, num_iter_standard, num_iter_learning, xs, xl

    def train(self,
              A, 
              s,
              eig_vec,
              model_prob,
              model_value,
              epochs,
              adam_decay_rate,
              lr,
              lr_decay_rate,
              lr_decay_step,
              device,
              single_model = False,
              enforce_pos = False,
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

        A = A.to(device).double()
        s = torch.Tensor(copy.deepcopy(s).squeeze()).reshape(9,1)
        eig_vec = eig_vec.to(device).double()
        stencil = torch.cat([s[0:4],s[5:9]]).t().to(device).double()
        LOSS = []
        assert stencil.shape[0]==1

        for epoch in range(epochs):
            loss = 0
            optimizer.zero_grad()
            prob = model_prob(stencil).squeeze()
            value = model_value(stencil).squeeze()
            coarse_stencil= self.sparsify(prob,value,single_model,softmax_on,enforce_pos=enforce_pos)

            temp = F.conv2d(eig_vec,coarse_stencil,padding=1)-A(eig_vec)
            temp = temp.squeeze(1).view(temp.shape[0],-1,1)
            loss = torch.norm(temp) ** 2
            LOSS.append(loss)
            loss.backward()
            # for name, param in model_value.named_parameters():
            #     print(name, param.grad)
            optimizer.step()
            scheduler.step()
            if verbose:
                if epoch % 100 == 0 or epoch==(epochs-1):
                    print(' epoch: ',epoch,' loss: ',loss)
        return model_prob, model_value, LOSS

    def get_stencil(self):
        a = np.random.uniform(low=1,high=2)
        b = np.random.uniform(low=2,high=3)
        c = np.random.uniform(low=0,high=1)
        s = np.array([[c,b,c],
                      [a,-2*(a+b)-4*c,a],
                      [c,b,c]])
        return s

    def optimal_stencil(self,s):
        s = s.squeeze()
        assert s.shape == (3,3)
        a = s[1,0]
        b = s[0,1]
        c = s[0,0]
        return np.array([[0,b+2*c,0],
                         [a+2*c,-2*(a+b)-8*c,a+2*c],
                         [0,b+2*c,0]])

    
    def generate_data(self,k:list,n:int):
        s = self.get_stencil()
        k1 = k[0]
        k2 = k[1]
        A = pyamg.gallery.stencil_grid(copy.deepcopy(s),(n,n)).tocsr()
        R = self.get_prolongation(n)
        P = R.T*4
        T = R @ P
        A = R @ A @ P

        _, eig_vec = sp.sparse.linalg.eigs(A,k=k1,M=T,which = 'SM')
        eig_vec = eig_vec.T
        eig_vec = torch.real(torch.from_numpy(eig_vec)).view(k1,1,n//2,n//2)

        # level 2 data
        n = n//2
        stencils = compute_stencil_numpy(A,n)
        s1 = stencils[n//2,n//2,:,:].T
        AA_test = pyamg.gallery.stencil_grid(s1,(n,n))
        assert np.linalg.norm(A.toarray()-AA_test.toarray())==0.0
        A_conv = nn.Conv2d(1, 1, 3, padding = 1, bias = False).double()
        A_conv.weight = nn.Parameter(torch.from_numpy(s1).view(1,1,3,3),requires_grad = False)

        # level 3 data
        R2 = self.get_prolongation(n)
        P2 = R2.T*4
        A2 = R2@A@P2
        T2 = R2 @ P2
        _, eig_vec2 = sp.sparse.linalg.eigs(A2,k=k2,M=T2,which = 'SM')
        eig_vec2 = eig_vec2.T
        eig_vec2 = torch.real(torch.from_numpy(eig_vec2)).view(k2,1,n//2,n//2)

        n = n//2
        stencils2 = compute_stencil_numpy(A2,n)
        s2 = stencils2[n//2,n//2,:,:].T
        A2_test = pyamg.gallery.stencil_grid(s2,(n,n))
        assert np.linalg.norm(A2.toarray()-A2_test.toarray()) == 0.0
        A2_conv = nn.Conv2d(1, 1, 3, padding = 1, bias = False).double()
        A2_conv.weight = nn.Parameter(torch.from_numpy(s2).view(1,1,3,3),requires_grad = False)

        return s, A_conv, A2_conv, s1, s2, eig_vec, eig_vec2

    def get_prolongation(self,grid_size):
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
                         coarse_solver='splu',
                         device=None,
                         single_model = False,
                         softmax_on=True,
                         enforce_pos = False,
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
            self.extend_hierarchy(levels,option1,models,device, single_model, softmax_on,enforce_pos=enforce_pos)

        ml = multilevel_solver(levels, coarse_solver=coarse_solver)
        change_smoothers(ml, presmoother, postsmoother)
        return ml

    def extend_hierarchy(self, levels, option1, models, device, single_model, softmax_on,enforce_pos=False):
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
        A_g = R@A@P
        A_g = A_g.astype(np.float64).tocsr()  # convert from complex numbers, should have A.imag==0
        levels[-1].Ag = A_g
        
        if option1=='standard':
            levels[-1].A = A_g

        elif option1=='learning':
            level_num = len(levels)
            model_key = f"level{level_num}"
            model_prob, model_value = models[model_key]
            model_prob.eval()
            model_value.eval()
            if n < 3:
                raise ValueError('Stencil at this level is not available because the size of the \
                                training grid is too small! Need to increase the \'n\'!')
            
            stencils = compute_stencil(A_g, n, n, 3)
            s = stencils[n//2,n//2,:,:]
            stencil = s.reshape(9,1)
            stencil = torch.from_numpy(stencil)
            stencil = torch.cat([stencil[0:4],stencil[5:9]]).t().to(device).double()
#             if level_num != 3:
            assert stencil.shape[0] == 1
            with torch.no_grad():
                prob = model_prob(stencil).squeeze()
                value = model_value(stencil).squeeze()
            learned_stencil = self.sparsify(prob,value,single_model,softmax_on,enforce_pos=enforce_pos)
            learned_stencil = learned_stencil.squeeze().detach().cpu().numpy()
#             learned_stencil = np.array([[ 0.00195787,0.,0.38254424],[ 0.72760527, -1.50085201, 0. ],[-0.,0., 0.38874463]])
            A_c = pyamg.gallery.stencil_grid(learned_stencil,(n,n))
            levels[-1].A = A_c
            levels[-1].s = learned_stencil
#             else:
#                 levels[-1].A = A_g
            # print('learning: ',A_c.count_nonzero()/A_c.shape[1])

        else:
            raise ValueError('Option1 is not available!')

    def sparsify(self, prob, value, single_model, softmax_on, enforce_pos=False):
#         if single_model:
#             # choose top values instead of top absolute values:
#             idx = torch.topk(value,4)[1]
#             val = torch.topk(value,4)[0]
#             # stencil = ...
#             # ...
#             # choose top absolute values:
#             # idx = torch.topk(torch.abs(value),4)[1]
#             # sign_val = torch.sign(value)[idx]
#             # val = torch.topk(torch.abs(value),4)[0]
#             # val = val * sign_val
#             stencil = torch.zeros_like(value,memory_format=torch.legacy_contiguous_format)
#             stencil = stencil.scatter_(-1,idx,val)
#             stencil = stencil - value.detach() + value
#             stencil = stencil.squeeze()
#             return torch.cat((stencil[0:4],-stencil.sum().view(1),stencil[4:8])).view(1,1,3,3)
#         else:
        if enforce_pos is None:
            stencil = top_k(prob,4,softmax_on).squeeze()
            stencil = stencil * value
            stencil = torch.cat((stencil[0:4],-stencil.sum().view(1),stencil[4:8])).view(1,1,3,3)
            return stencil
        elif enforce_pos == '+':
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
            return stencil
        elif enforce_pos == 'x':
            raise KeyError(f"{enforce_pos} has not been implemented yet")