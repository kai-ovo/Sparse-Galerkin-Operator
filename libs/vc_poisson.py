from types import NoneType
from .pde import *
from .utils import *
import time
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy as sp
from scipy.sparse import find
import copy
from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
from scipy.sparse import csr_matrix, isspmatrix_csr

class VarCoeffPoisson(PDEData):
    def __init__(self, grid_size, k,
                 aparams=(0.1,'test') ):
        """
        Construct a problem -div( a(x,y)*grad(u(x,y)) ) = f(x,y)
        grid_size : training mesh size (level 1 size; sparsification starts on level 2)
        aparams : parameters that define a (x)
                  aparams = (kappa, coefficient_function_option)
        """
        super().__init__()
        self.n = grid_size
        self.kappa, self.coeff_opt = aparams
        self.k1, self.k2 = k[0],k[1]
        """
        stencils: (n^2, stencil size)
        E: edge index, torch.tensor
        evec: smooth vectors
        A: galerkin coarse operator
        """
        self.stencils2, self.E2, self.evec2, self.A2, self.stencils3, self.E3, self.evec3, self.A3 = \
            self.generate_data(self.n,lambda x,y : self.a(x,y,kappa=self.kappa, option=self.coeff_opt))
        self.stencils2 = torch.cat((self.stencils2[:,:4],self.stencils2[:,-4:]),dim=-1)
        self.stencils3 = torch.cat((self.stencils3[:,:4],self.stencils3[:,-4:]), dim=-1)
    
    def train(self,
              A_g, 
              stencils, 
              edge_index,
              eig_vec,
              GAT_prob, GAT_value,
              epochs,
              adam_decay_rate,
              lr,
              lr_decay_rate,
              lr_decay_step,
              device,
              num_nhbr = 5,
              single_model = False,
              enforce_stencil_symmetry = None,
              softmax_on = True,
              verbose = False):

        if single_model:
            GAT_value = GAT_value.to(device).double()
            optimizer = torch.optim.Adam(list(GAT_value.parameters()), lr=lr, 
                                        weight_decay=adam_decay_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step,
                                                        gamma=lr_decay_rate)
        else:
            GAT_prob = GAT_prob.to(device).double()
            GAT_value = GAT_value.to(device).double()
            optimizer = torch.optim.Adam(list(GAT_prob.parameters())+list(GAT_value.parameters()), lr=lr, 
                                         weight_decay=adam_decay_rate)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step,
                                                    gamma=lr_decay_rate)
        # model_prob = model_prob.to(device).double()
        # model_value = model_value.to(device).double()
        # optimizer = torch.optim.Adam(list(model_prob.parameters())+list(model_value.parameters()), 
                                    # lr=lr, 
                                    # weight_decay=adam_decay_rate)
        # optimizer = torch.optim.Adam(list(GAT_prob.parameters())+list(GAT_value.parameters()), lr=lr, 
        #                              weight_decay=adam_decay_rate)
        
        num_vec = eig_vec.shape[1]
        if type(A_g) is not torch.Tensor:
            if type(A_g) is not np.ndarray:
                A_g = A_g.toarray()
            A_g = torch.from_numpy(A_g).to_sparse().to(device)
        else:
            A_g = A_g.to_sparse().to(device)
        
        eig_vec = torch.from_numpy(eig_vec.copy()).to(device).double()
        stencils = stencils.clone().to(device).double()
        edge_index = edge_index.clone().to(device)

        for epoch in range(epochs):
            loss = 0
            optimizer.zero_grad()
            # GAT returns a non sparsified stencil
            gat_stencils = None
            prob = None
            value = None

            # forward pass
            if single_model:
                gat_stencils = GAT_value(stencils, edge_index)
            else:
                prob = GAT_prob(stencils, edge_index)
                value = GAT_value(stencils, edge_index)
            
            # sparsification
            sparsified_stencils = self.sparsify(gat_stencils, prob, value, softmax_on,num_nhbr=num_nhbr,single_model=single_model)
            assert stencils.shape == sparsified_stencils.shape

            # reorder stencils
            S1 = sparsified_stencils[:,:4]
            S2 = -sparsified_stencils.sum(dim=-1).reshape(-1,1)
            S3 = sparsified_stencils[:,-4:]
            learned_sparsified_stencils = torch.cat((S1,S2,S3), dim=-1)
            d = int(sqrt(stencils.size(0)))

            # reshape stencils
            learned_sparsified_stencils = learned_sparsified_stencils.reshape(d,d,3,3)

            # compute A_c
            A_c = compute_A_torch(learned_sparsified_stencils, d)

            # compute loss
            for j in range(num_vec):
                v1 = torch.sparse.mm(A_c,eig_vec[:,j].reshape(-1,1).to(device))
                v2 = torch.sparse.mm(A_g,eig_vec[:,j].reshape(-1,1).to(device))
                temp = v1-v2
                loss += torch.norm(temp)
            loss = loss / num_vec
            loss.backward()
            # for name, param in model_value.named_parameters():
            #     print(name, param.grad)
            optimizer.step()
            scheduler.step()
            if verbose:
                if epoch % 100 == 0 or epoch==(epochs-1):
                    print(' epoch: ',epoch,' loss: ',loss)
        if single_model:
            return GAT_value, None
        else:
            return GAT_prob, GAT_value
    
    def train_attn(self,
                   A_g, 
                   stencils, 
                   eig_vec,
                   model_prob, model_value,
                   epochs,
                   adam_decay_rate,
                   lr,
                   lr_decay_rate,
                   lr_decay_step,
                   device,
                   num_nhbr = 5,
                   single_model = False,
                   enforce_stencil_symmetry : str = None, # '+' or 'x'
                   softmax_on = True,
                   verbose = False):

        if single_model:
            model_value = model_value.to(device).double()
            optimizer = torch.optim.Adam(list(model_value.parameters()), lr=lr, 
                                        weight_decay=adam_decay_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step,
                                                        gamma=lr_decay_rate)
        else:
            model_prob = model_prob.to(device).double()
            model_value = model_value.to(device).double()
            optimizer = torch.optim.Adam(list(model_prob.parameters())+list(model_value.parameters()), lr=lr, 
                                         weight_decay=adam_decay_rate)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step,
                                                    gamma=lr_decay_rate)

        num_vec = eig_vec.shape[1]
        if type(A_g) is not torch.Tensor:
            if type(A_g) is not np.ndarray:
                A_g = A_g.toarray()
            A_g = torch.from_numpy(A_g).to_sparse().to(device)
        else:
            A_g = A_g.to_sparse().to(device)
        
        eig_vec = torch.from_numpy(eig_vec.copy()).to(device).double()
        stencils = stencils.clone().to(device).double()

        for epoch in range(epochs):
            loss = 0
            optimizer.zero_grad()
            # GAT returns a non sparsified stencil
            gat_stencils = None
            prob = None
            value = None

            # forward pass
            if single_model:
                gat_stencils = model_value(stencils).squeeze()
            else:
                prob = model_prob(stencils).squeeze()
                value = model_value(stencils).squeeze()
            
            # sparsification
            sparsified_stencils = self.sparsify(gat_stencils, prob, value, softmax_on,num_nhbr=num_nhbr,single_model=single_model)
            assert stencils.shape == sparsified_stencils.shape

            # reorder stencils
            S1 = sparsified_stencils[:,:4]
            S2 = -sparsified_stencils.sum(dim=-1).reshape(-1,1)
            S3 = sparsified_stencils[:,-4:]
            learned_sparsified_stencils = torch.cat((S1,S2,S3), dim=-1)
            d = int(sqrt(stencils.size(0)))

            # reshape stencils
            learned_sparsified_stencils = learned_sparsified_stencils.reshape(d,d,3,3)

            # compute A_c
            A_c = compute_A_torch(learned_sparsified_stencils, d)

            # compute loss
            for j in range(num_vec):
                v1 = torch.sparse.mm(A_c,eig_vec[:,j].reshape(-1,1).to(device))
                v2 = torch.sparse.mm(A_g,eig_vec[:,j].reshape(-1,1).to(device))
                temp = v1-v2
                loss += torch.norm(temp)
            loss = loss / num_vec
            loss.backward()
            # for name, param in model_value.named_parameters():
            #     print(name, param.grad)
            optimizer.step()
            scheduler.step()
            if verbose:
                if epoch % 100 == 0 or epoch==(epochs-1):
                    print(' epoch: ',epoch,' loss: ',loss)
        if single_model:
            return model_value, None
        else:
            return model_prob, model_value

    def test_model(self,
                   models : dict,
                   test_grid_size : int,
                   device,
                   top_accel = None,
                   max_levels : int = 2, 
                   num_nhbr = 5,
                   maxits : int = 1000,
                   tol = 1e-6,
                   softmax_on = False,
                   single_model = True,
                   enforce_stencil_symmetry=None,
                   use_attention = True,
                   verbose = True):
        """
        models['level{k}'] = sparsification models on level k
        """
        res_standard = []
        res_learning = []
        stencils = self.get_problem(test_grid_size,lambda x,y : self.a(x,y,kappa=self.kappa, 
                                                                       option=self.coeff_opt))
        A = compute_A_sparse(stencils, test_grid_size)
        solver_standard = self.geometric_solver(A,
                                                'standard',
                                                models,
                                                test_grid_size,
                                                num_nhbr = num_nhbr,
                                                max_levels = max_levels,
                                                single_model = single_model,
                                                attention = use_attention,
                                                softmax_on = softmax_on,
                                                coarse_solver = 'splu')
                                                
        solver_non_galerkin = self.geometric_solver(A,
                                                    'learning',
                                                    models,
                                                    test_grid_size,
                                                    num_nhbr = num_nhbr,
                                                    max_levels = max_levels,
                                                    coarse_solver = 'splu',
                                                    single_model = single_model,
                                                    attention = use_attention,
                                                    softmax_on = softmax_on,
                                                    device = device)
        x0 = np.ones((A.shape[0],1))
        b = np.random.rand(A.shape[0],1)
        
        t1 = time.time()
        xs = solver_standard.solve(b,x0=x0,maxiter=maxits, tol=tol,residuals=res_standard, accel=top_accel)
        t2 = time.time()
        xl = solver_non_galerkin.solve(b,x0=x0,maxiter=maxits, tol=tol,residuals=res_learning, accel=top_accel)
        t3 = time.time()

        num_iter_standard = len(res_standard)
        num_iter_learning = len(res_learning)
        ts = t2 - t1
        tl = t3 - t2
        if verbose:
            print('standard iteration:  ', num_iter_standard,
                  ' standard time:  ', ts)
            print('leanred iteration:   ', num_iter_learning,
                  ' learned time:   ', tl)
        
        return num_iter_standard, num_iter_learning, xs, xl, res_standard, res_learning

    def geometric_solver(self, A,
                         mg_option : str,  
                         models,
                         n, # grid size
                         presmoother=('gauss_seidel', {'sweep': 'forward'}),
                         postsmoother=('gauss_seidel', {'sweep': 'forward'}),
                         num_nhbr = 5,
                         max_levels = 3, 
                         max_coarse = 10,
                         coarse_solver='splu',
                         device = None,
                         attention=True,
                         single_model = False,
                         softmax_on=True,
                         **kwargs):
        """
        models - dictionary
                 models['level{k}'] contains trained models for sparsification on level k
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

        levels[-1].A = copy.deepcopy(A)
        levels[-1].Ag = copy.deepcopy(A)
        levels[-1].n = n

        while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
            self.extend_hierarchy(levels,mg_option,models,device, softmax_on, num_nhbr,single_model=single_model, attention=attention)

        ml = multilevel_solver(levels, coarse_solver=coarse_solver)
        change_smoothers(ml, presmoother, postsmoother)
        return ml
    
    def extend_hierarchy(self, levels, mg_option, models, device, softmax_on, num_nhbr, single_model=False, attention=True):
        """Extend the multigrid hierarchy."""

        A = copy.deepcopy(levels[-1].Ag)
        n = levels[-1].n

        # prolongtion and restriction
        R = self.get_prolongation(n)
        P = R.T*4
        levels[-1].P = P  # prolongation operator
        levels[-1].R = R  # restriction operator
        levels.append(multilevel_solver.level()) # add a new level
        n = n//2
        levels[-1].n = n
        A_g = R@A@P # convert from complex numbers, should have A.imag==0
        A_g = A_g.astype(np.float64).tocsr()
        levels[-1].Ag = A_g

        if mg_option=='standard': 
            levels[-1].A = A_g
            # print('standard: ', A.count_nonzero() / A.shape[1])

        elif mg_option=='learning':
            # if len(levels) == 1:
            level_num = len(levels)
            model_key = f"level{level_num}"

            # load model(s)
            if single_model:
                model = models[model_key].to(device)
                model.eval()
            else:
                model_prob, model_value = models[model_key]
                model_prob, model_value = model_prob.to(device), model_value.to(device)
                model_prob.eval()
                model_value.eval()

            if n < 3:
                raise ValueError('Stencil at this level is not available because the size of the \
                                training grid is too small! Need to increase the \'n\'!')
            stencils = compute_stencil_numpy(A_g,n)
            stencils = torch.from_numpy(stencils).reshape(n**2, -1).to(device)
            concat_stencils = torch.cat((stencils[:,:4],stencils[:,-4:]),dim=-1)
            E = self.get_edge_index(A_g).to(device)

            # evaluate model
            gat_stencils = None
            prob = None
            value = None
            with torch.no_grad():
                if single_model:
                    if not attention:
                        gat_stencils = model(concat_stencils, E)
                    else:
                        gat_stencils = model(concat_stencils).squeeze()
                else:
                    if not attention:
                        prob = model_prob(concat_stencils,E)
                        value = model_value(concat_stencils,E)
                    else:
                        prob = model_prob(concat_stencils).squeeze()
                        value = model_value(concat_stencils).squeeze()

            # sparsification
            sparsified_stencils = self.sparsify(gat_stencils,prob,value,softmax_on,num_nhbr=num_nhbr, single_model=single_model)

            # regroup
            A1 = sparsified_stencils[:,:4]
            A2 = -sparsified_stencils.sum(dim=-1).reshape(-1,1)
            A3 = sparsified_stencils[:,-4:]
            sparsified_stencils = torch.cat((A1,A2,A3), dim=-1)
            assert stencils.shape == sparsified_stencils.shape

            d = int(sqrt(stencils.size(0)))
            sparsified_stencils = sparsified_stencils.reshape(d,d,3,3)

            A_c = compute_A_sparse(sparsified_stencils.detach().cpu().numpy(),n)
            levels[-1].A = A_c

        else:
            raise ValueError('Option1 is not available!')

    def generate_data(self, n, a):
        """
        n: mesh size
        a: coefficient function
        """
        # level 1 stencils
        stencils1 = self.get_problem(n, a) 
        A = compute_A_sparse(stencils1,n)

        # level 2 
        R2 = self.get_prolongation(n)
        P2 = R2.T                             
        A2 = R2@A@P2
        T2 = R2 @ P2
        _, eig_vec2 = sp.sparse.linalg.eigs(A2,k=self.k1,M=T2,which = 'SM')
        n = n//2

        stencils2 = compute_stencil_numpy(A2,n)
        stencils2 = torch.from_numpy(stencils2).reshape(n**2,-1)
        E2 = self.get_edge_index(A2)

        # level 3
        R3 = self.get_prolongation(n)
        P3 = R3.T
        T3 = R3 @ P3
        A3 = R3@A2@P3
        _, eig_vec3 = sp.sparse.linalg.eigs(A3,k=self.k2,M=T3,which = 'SM')
        n = n//2

        stencils3 = compute_stencil_numpy(A3,n)
        stencils3 = torch.from_numpy(stencils3).reshape(n**2,-1)
        E3 = self.get_edge_index(A3)

        return stencils2, E2, eig_vec2, A2, stencils3, E3, eig_vec3, A3
    
    
    def a(self, x, y, kappa=0.1, option='test'):
        if option == 'toy':
            return x
        elif option == 'Toy':
            return x+y
        elif option == 'TOY':
            return 0.1*x+3*y+1.5*y**2
        elif option == 'test':
            return np.sin(kappa*np.pi*x*y)+1.01

    def get_edge_index(self, A) -> torch.Tensor:
        """
        Build 2-D mesh edge index when using 9-point stencil
        
        Input:
                A: stiffness matrix (sparse)
                
        Output:
                E: a (2,|E|) matrix
                E[:,j] contains the indices of the end nodes of an edge
        
        """
        B = copy.deepcopy(A)
        B.setdiag(0) # remove self edge
        e1 = np.asarray(find(B)[1])
        e2 = np.asarray(find(B)[0])
        return torch.from_numpy(np.vstack((e1,e2)))

    def get_problem(self, n, a) -> np.ndarray:
        """
        9-point stencils
        Input:
                n: mesh size
                a: coefficient function a(x,y)
        Output: 
                (n, n, 3, 3) matrix S
                S[i,:] contains the stencil information at point i
        """

        x = [(i+1)/(n+1) for i in range(n)]
        y = x
        stencil = np.zeros((n,n,3,3))
        h = 1/(n+1)
        for i in range(n):
            for j in range(n):
                # Compute each stencil
                xnw = (x[i-1]+x[i])/2 if i-1>=0 else x[i]
                ynw = (y[j+1]+y[j])/2 if j+1<n else y[j]
                xne = (x[i+1]+x[i])/2 if i+1<n else x[i]
                yne = (y[j+1]+y[j])/2 if j+1<n else y[j]
                xsw = (x[i-1]+x[i])/2 if i-1>=0 else x[i]
                ysw = (y[j-1]+y[j])/2 if j-1>=0 else y[j]
                xse = (x[i+1]+x[i])/2 if i+1<n else x[i]
                yse = (y[j-1]+y[j])/2 if j-1>=0 else y[j ]
                gnw = a(xnw,ynw)
                gne = a(xne,yne)
                gsw = a(xsw,ysw)
                gse = a(xse,yse)
                
                stencil[i,j,0,2] =  -1/3*gnw
                stencil[i,j,1,2] = -1/6*(gnw+gne)
                stencil[i,j,2,2] = -1/3*gne
                stencil[i,j,0,1] = -1/6*(gsw+gnw)
                stencil[i,j,1,1] = 2/3*(gnw+gne+gse+gsw)
                stencil[i,j,2,1] = -1/6*(gne+gse)
                stencil[i,j,0,0] = -1/3*gsw
                stencil[i,j,1,0] = -1/6*(gse+gsw)
                stencil[i,j,2,0] = -1/3*gse
        stencil[ :, 0, :, 0] = 0.
        stencil[ :, -1, :, -1] = 0.
        stencil[ 0, :, 0, :] = 0.
        stencil[ -1, :, -1, :] = 0.
        return stencil
    
    def sparsify(self, stencils, prob, value, softmax_on, num_nhbr=5, single_model=False):
        """
        prob.shape = (n^2, -1)
        r : num points in the stencil
        """
           
        # if softmax_on:
        #     prob = prob.softmax(-1)
        

        # single model implementation
        # choose top absolute values implementation
#         idx = torch.topk(torch.abs(stencils),num_nhbr, dim=-1)[1]
#         idx_help = torch.arange(idx.size(0)).reshape(-1,1).repeat(1,num_nhbr)
#         assert idx.shape == idx_help.shape
#         sign_val = torch.sign(stencils)[idx_help, idx]
#         val = torch.topk(torch.abs(stencils.clone()),num_nhbr, dim=-1)[0]
#         val = val * sign_val

        # choose top values implementation
        if single_model:
            assert prob is None and value is None
            if softmax_on:
                stencils = stencils.softmax(-1)
            idx = torch.topk(stencils,num_nhbr, dim=-1)[1]
            val = torch.topk(stencils,num_nhbr, dim=-1)[0]
            stencil = torch.zeros_like(stencils, memory_format=torch.legacy_contiguous_format)

            stencil = stencil.scatter_(-1,idx,val)
            stencil = stencil - stencils.detach() + stencils
            stencil = stencil.squeeze()
        else:
            assert stencils is None
            if softmax_on:
                prob = prob.softmax(-1)
            idx = torch.topk(prob, num_nhbr, dim=-1)[1]
            mask = torch.zeros_like(value, memory_format=torch.legacy_contiguous_format)
            mask = mask.scatter_(-1,idx,1.0)
            stencil = value * mask
            stencil = stencil - value.detach() + value

        # double model implementation
        # idx = torch.topk(prob, num_nhbr, dim=-1)[1]
        # mask = torch.zeros_like(value, memory_format=torch.legacy_contiguous_format)
        # mask = mask.scatter_(-1,idx,1)
        # mask = mask - prob.detach() + prob
        # stencil = value * mask
        # # print('sparsified stencil:', stencil)
        # stencil = stencil.squeeze()
        return stencil
        

    # def sparsify(self, prob, value, single_model, softmax_on, r=5):
    #     """
    #     prob.shape = (n^2, -1)
    #     r : num points in the stencil
    #     """
    #     # TODO: check the correctness of topk_dim (whether scatter gives the desired mask matrix)
    #     # TODO: check the correctness of single model implementation herein and in topk
    #     stencil = topk_dim(prob,r,softmax_on,single_model,dim=-1).squeeze()   
    #     if not single_model:
    #         stencil = stencil * value
    #         # print(stencil)
    #         stencil = torch.cat((stencil[0:4],-stencil.sum(dim=-1).view(1),stencil[4:8])).view(1,1,3,3)
    #     else:
    #         stencil = torch.cat((stencil[0:4],-stencil.sum().view(1),stencil[4:8])).view(1,1,3,3)
    #     return stencil
    
    def get_prolongation(self,grid_size):
        """
        full coarsening prolongation operator
        """
        res_stencil = np.double(np.zeros((3,3)))
        k=16
        res_stencil[0,0] = 1/k
        res_stencil[0,1] = 2/k
        res_stencil[0,2] = 1/k
        res_stencil[1,0] = 2/k
        res_stencil[1,1] = 4/k
        res_stencil[1,2] = 2/k
        res_stencil[2,0] = 1/k
        res_stencil[2,1] = 2/k
        res_stencil[2,2] = 1/k
        P_stencils= np.zeros((grid_size//2,grid_size//2,3,3))
        for i in range(grid_size//2):
            for j in range(grid_size//2):
                P_stencils[i,j,:,:]=res_stencil
        return compute_p2_numpy(P_stencils, grid_size).astype(np.double)
    
    