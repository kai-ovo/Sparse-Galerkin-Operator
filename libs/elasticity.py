from audioop import alaw2lin
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


class Elasticity(PDEData):
    def __init__(self, train_grid_size : int, E, nu, k:list, dataset_option='fixed E'):
        """
        train_grid_size : int
            training grid size
        E : double or list
            Young's modulus
        nu : double or list
            Poisson's ratio
        k : list
            k[0] : number of training vectors on level 2
            k[1] : number of training vectors on level 3
        dataset_option : str
            if 'fixed E', then nu must be a list
        """
        self.n = train_grid_size
        self.E = E
        self.nu = nu
        self.k = k
        self.dataset_option = dataset_option
        self.Auu2_train, self.Auv2_train, self.s2_train, self.vec_uu2_train, self.vec_vv2_train = \
                            self.generate_data(self.E, self.nu, self.n, self.k, dataset_option=self.dataset_option)

    def generate_data(self, EList, nuList, n, k, dataset_option='fixed E'):
        """
        TODO: finish level-3 data generating pipeline
        """
        # type casting
        k2 = k[0]
        if type(EList) is not list:
            EList = list(EList)
        if type(nuList) is not list:
            nuList = list(nuList)
        
        # shape check
        if dataset_option == 'fixed E':
            assert len(EList) == 1
            npts = len(nuList)
            EList = EList * npts
        elif dataset_option == 'fixed nu':
            assert len(nuList) == 1
            npts = len(EList)
            nuList = nuList * npts
        else:
            raise KeyError("Dataset Option is not available")
        
        params = zip(EList, nuList)

        Auu2_train = []
        Auv2_train = []
        s2_train = []
        vec_uu2_train = []
        vec_vv2_train = []
        for param in params:
            E, nu = param
            Auu, Auv, s, vec_uu, vec_vv = self.gen_data_point(E,nu,n,k2)
            Auu2_train.append(Auu)
            Auv2_train.append(Auv)
            s2_train.append(s)
            vec_uu2_train.append(vec_uu)
            vec_vv2_train.append(vec_vv)
        
        return Auu2_train, Auv2_train, s2_train, vec_uu2_train, vec_vv2_train

    def train(self, 
              Auu_train, 
              Auv_train, 
              vec_uu_train, 
              vec_vv_train, 
              s_train,  
              model_prob, # model
              model_value, # model
              epochs, 
              lr, 
              lr_decay_rate,
              lr_decay_step,
              adam_decay_rate,
              device,
            #   single_model = False,
            #   softmax_on = True,
              verbose = True):
              
        model_prob = model_prob.to(device).double()
        model_value = model_value.to(device).double()
        optimizer = torch.optim.Adam(list(model_prob.parameters())+list(model_value.parameters()), lr=lr, 
                                     weight_decay=adam_decay_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=lr_decay_step,
                                                    gamma=lr_decay_rate)
        LOSS = []

        for epoch in range(epochs):
            loss = 0
            optimizer.zero_grad()
        
            for j in range(len(Auv_train)):
                # for j in range(len(A_train1)):
                # s_uu,s_uv,s_vu,s_vv = s_train
                s_uu,s_uv,s_vu,s_vv = s_train[j]
                A_uu_conv = Auu_train[j]
                A_uv_conv = Auv_train[j]  
                test_vec_uu = vec_uu_train[j]
                test_vec_vv = vec_vv_train[j]
                n = 17 # TODO: check what this n = 17 corresponds to and rewrite it according to available variables

                # s_uv = torch.from_numpy(s_uv).view(-1,1).clone()
                # s_vu = torch.from_numpy(s_vu).view(-1,1).clone()
                # s_vv = torch.from_numpy(s_vv).view(-1,1).clone()
                # stencil_uu_train = torch.cat([s_uu[0:24],s_uu[25:49]])
                # stencil_uv_train = torch.cat([s_uv[0:24],s_uv[25:49]])
                # stencil_vu_train = torch.cat([s_vu[0:24],s_vu[25:49]])
                # stencil_vv_train = torch.cat([s_vv[0:24],s_vv[25:49]])
                # stencil = torch.cat([stencil_uu_train,stencil_uv_train,stencil_vu_train,stencil_vv_train]).view(1,-1)
                stencil = torch.zeros(1,6).double()
                stencil[0,0] = s_uu[0,1]
                stencil[0,1] = s_uu[0,3]
                stencil[0,2] = s_uu[1,1]
                stencil[0,3] = s_uu[1,2]
                stencil[0,4] = s_uu[1,3]
                stencil[0,5] = s_uv[0,1]
                # stencil[0,6] = s_uv[1,1]
                # stencil[0,7] = s_uv[1,2]
                # stencil[0,8] = s_uv[2,1]
                # stencil[0,9] = s_uv[2,2]
                # stencil[0,10] = s_vu[1,1]
                # stencil[0,11] = s_vu[1,2]
                # stencil[0,12] = s_vu[2,1]
                # stencil[0,13] = s_vu[2,2]

                # stencil[0,7] = s_uu[3,3]
                # stencil[0,8] = s_uu[3,4]
                # stencil[0,9] = s_uu[3,5]
                # stencil[0,10] = s_uu[4,1]
                # stencil[0,11] = s_uu[4,2]
                # stencil[0,12] = s_uu[4,3]
                # stencil[0,13] = s_uu[4,4]
                # stencil[0,14] = s_uu[4,5]

                prob = model_prob(stencil).squeeze()
                # print(stencil)
                value = model_value(stencil).squeeze()
                # print(value)
                # print(stencil)
                # value = stencil
                stencils, s, ss = self.parsify(prob,value,int(len(prob)*0.5),stencil_train)
                s_uu,s_uv,s_vu,s_vv = s
                ss_uu,ss_uv,ss_vu,ss_vv = ss
                stencil_uu,stencil_uv,stencil_vu,stencil_vv = stencils

                temp_uu = F.conv2d(test_vec_uu,s_uu.view(1,1,5,5),padding=0)-A_uu_conv(test_vec_uu)
                temp_uv = F.conv2d(test_vec_vv,s_uv.view(1,1,4,4),padding=0)-A_uv_conv(test_vec_vv)

                # temp = ss_uu-A_uu_conv.weight.squeeze()
                loss += torch.norm(temp_uu)**2+torch.norm(temp_uv)**2
            loss.backward()
            optimizer.step()
            scheduler.step()
            LOSS.append(loss)

            # temp = torch.mm(A_c,eig_vec)-torch.mm(Ag.to_dense(),eig_vec)
            # # print(temp)
            # # temp = temp.squeeze(1).view(temp.shape[0],-1,1)
            # loss+=torch.norm(temp)**2
            # loss.backward()

            # for name, param in model_value.named_parameters():
            #     print(name, param.grad)
            if verbose:
                if epoch % 10 == 0 or epoch == (epochs-1):
                    print('epoch:   ',epoch,'   loss:   ',loss)

        A_uu = compute_A(stencil_uu,stencil_uu.shape[0],stencil_uu.shape[1],stencil_uu.shape[2]).to_dense()
        A_uv = compute_A(stencil_uv,stencil_uv.shape[0],stencil_uv.shape[1],stencil_uv.shape[2]).to_dense()
        A_vu = compute_A(stencil_vu,stencil_vu.shape[0],stencil_vu.shape[1],stencil_vu.shape[2]).to_dense()
        A_vv = compute_A(stencil_vv,stencil_vv.shape[0],stencil_vv.shape[1],stencil_vv.shape[2]).to_dense()
        A_uu_uv = torch.hstack((A_uu,A_uv))
        A_vu_vv = torch.hstack((A_vu,A_vv))
        A_c = torch.vstack((A_uu_uv,A_vu_vv))
        A_c = A_c[0:n*n*2:2,:]
        A_c = A_c[:,0:n*n*2:2]

        return model_prob, model_value, A_c, ss_uu, ss_vv, ss_uv, ss_vu, LOSS

    def test_model(self):
        pass

    def gen_data_point(self, E, nu, n, k):
        """
        E: Young's modulus
        nu: poisson's ratio
        n: mesh size

        k: number of training vectors
            k[0]: number of smooth vectors on level 2
            k[1]: number of smooth vectors on level 3
            ...
        TODO: finish

        """
        k2, k3 = k[0], k[1] # number of smooth vectors
        A = self.get_problem(n,E,nu)
        P2, R2, Pf2, Rf2 = self.get_prolongation(n)
        T2 = Rf2 @ Pf2
        A2 = Rf2 @ A @ Pf2
        # confusion in Ru's code: Do not run this for now. why R_train???? why not R??
        assert 0==1
        A_uu = copy.deepcopy(A2[:n*n,:n*n])
        A_uv = copy.deepcopy(A2[:n*n,n*n:])
        A_vu = copy.deepcopy(A2[n*n:,:n*n])
        A_vv = copy.deepcopy(A2[n*n:,n*n:])

        stencil_uu = compute_stencil(A_uu,n,n,7)
        stencil_uv = compute_stencil(A_uv,n,n,5)
        stencil_vu = compute_stencil(A_vu,n,n,5)
        stencil_vv = compute_stencil(A_vv,n,n,7)

        s_uu = stencil_uu[n//2,n//2,:,:]
        s_uu[abs(s_uu)<1e-15] = 0
        s_uv = stencil_uv[n//2,n//2,:,:]
        s_uv[abs(s_uv)<1e-15] = 0
        s_vu = stencil_vu[n//2,n//2,:,:]
        s_vu[abs(s_vu)<1e-15] = 0
        s_vv = stencil_vv[n//2,n//2,:,:]
        s_vv[abs(s_vv)<1e-15] = 0

        #######
        # TODO: check what reorder does
        #######
        s_uu, _, _ = reorder(s_uu,1)
        s_uv, _, _ = reorder(s_uv,2)
        s_vu, _, _ = reorder(s_vu,2)
        s_vv, _, _ = reorder(s_vv,1)
        
        A_uu_conv = nn.Conv2d(1, 1, 3, padding = 0, bias = False).double()
        s_uu = s_uu[1:6,1:6]
        A_uu_conv.weight = nn.Parameter(s_uu.view(1,1,5,5))
        A_vv_conv = nn.Conv2d(1, 1, 3, padding = 0, bias = False).double()
        s_vv = s_vv[1:6,1:6]
        A_vv_conv.weight = nn.Parameter(s_vv.view(1,1,5,5))

        A_uv_conv = nn.Conv2d(1, 1, 4, padding = 0, bias = False).double()
        s_uv = s_uv[0:4,0:4]
        A_uv_conv.weight = nn.Parameter(s_uv.view(1,1,4,4))
        A_vu_conv = nn.Conv2d(1, 1, 4, padding = 0, bias = False).double()
        s_vu = s_vu[0:4,0:4]
        A_vu_conv.weight = nn.Parameter(s_vu.view(1,1,4,4))

        # stencil_train = [stencil_uu,stencil_uv,stencil_vu,stencil_vv]
        s = [s_uu,s_uv,s_vu,s_vv]
        _, eig_vec = sp.sparse.linalg.eigs(A2, M=T2, k=k2, which = 'SM')
        eig_vec = eig_vec.T
        eig_vec_uu = eig_vec[:,0:n*n].reshape(k2,n,n)
        eig_vec_vv = eig_vec[:,n*n:].reshape(k2,n,n)
        eig_vec_uu = torch.real(torch.from_numpy(eig_vec_uu))
        eig_vec_vv = torch.real(torch.from_numpy(eig_vec_vv))
        test_vec_uu = reorder(eig_vec_uu,1)[0].view(k2,1,n,n)
        test_vec_vv = reorder(eig_vec_vv,1)[0].view(k2,1,n,n)
        return A_uu_conv, A_uv_conv, s, test_vec_uu, test_vec_vv


    def get_stencil(self,E=1e-5,nu=0.3,by=1):
        """
        generate stencil for one instance

        code verified
        """
        s1,s2 = elasticity(E,nu,by)
        return s1,s2

    def get_prolongation(self, n):
        """
        prolongation operator

        code verified
        """
        res_uu = np.zeros((3,3))
        res_uu[0,1] = .25
        res_uu[1,0] =  .25
        res_uu[1,1] =  1
        res_uu[1,2] =  .25
        res_uu[2,1] =  .25

        res_uv = np.zeros((3,3))
        res_uv[0,1] = .25
        res_uv[1,0] = -.25
        res_uv[1,1] = 0
        res_uv[1,2] = -.25
        res_uv[2,1] = .25

        res_vu = np.zeros((3,3))
        res_vu[0,1] = -.25
        res_vu[1,0] = .25
        res_vu[1,1] = 0
        res_vu[1,2] = .25
        res_vu[2,1] = -.25

        res_vv = np.zeros((3,3))
        res_vv[0,1] =.25
        res_vv[1,0] = .25
        res_vv[1,1] = 1
        res_vv[1,2] =.25
        res_vv[2,1] = .25

        R_uu = pyamg.gallery.stencil_grid(res_uu,(n,n)).tocsr() # (n^2,n^2)
        R_uv = pyamg.gallery.stencil_grid(res_uv,(n,n)).tocsr() # (n^2,n^2)
        R_vu = pyamg.gallery.stencil_grid(res_vu,(n,n)).tocsr() # (n^2,n^2)
        R_vv = pyamg.gallery.stencil_grid(res_vv,(n,n)).tocsr() # (n^2,n^2)

        R_uu_uv = sp.sparse.hstack((R_uu,R_uv)) # (n^2, 2*n^2)
        R_vu_vv = sp.sparse.hstack((R_vu,R_vv)) # (n^2, 2*n^2)
        Rf = sp.sparse.vstack((R_uu_uv,R_vu_vv)).tocsr() # (2*n^2, 2*n^2)
        Pf = Rf.T # (2*n^2, 2*n^2)
        R = Rf[0:n*n*2:2,:] # (n^2, 2*n^2)
        P = R.T # (2*n^2, n^2)
        return P, R, Pf, Rf

    def get_problem(self,n,E=1e-5,nu=0.3):
        """
        generate level 1 linear system

        code verified
        """
        s1,s2 = self.get_stencil(E,nu)
        s3 = transpose_stencil_numpy(copy.deepcopy(s2))
        s4 = transpose_stencil_numpy(copy.deepcopy(s1))
        Auu = pyamg.gallery.stencil_grid(s1,(n,n))
        Auv = pyamg.gallery.stencil_grid(s2,(n,n))
        Avu = pyamg.gallery.stencil_grid(s3,(n,n))
        Avv = pyamg.gallery.stencil_grid(s4,(n,n))

        A_uu_uv = sp.sparse.hstack((Auu,Auv))
        A_vu_vv = sp.sparse.hstack((Avu,Avv))
        A = sp.sparse.vstack((A_uu_uv,A_vu_vv))
        return A
    
    def geometric_solver(self, A, mg_option : str, models : dict, n : int,
                         presmoother=('gauss_seidel', {'sweep': 'forward'}),
                         postsmoother=('gauss_seidel', {'sweep': 'forward'}),
                         max_levels=2, max_coarse=10, coarse_solver='splu'):
   
        levels = [multilevel_solver.level()]

        # convert A to csr
        if not isspmatrix_csr(A):
            try:
                A = csr_matrix(A)
                # warn("Implicit conversion of A to CSR",
                #     SparseEfficiencyWarning)
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
            self.extend_hierarchy(levels, mg_option, models)

        ml = multilevel_solver(levels, coarse_solver=coarse_solver)
        change_smoothers(ml, presmoother, postsmoother)
        return ml

    def extend_hierarchy(self, levels, mg_option, models : dict):
        """
        Extend the multigrid hierarchy.
        """
        A = levels[-1].Ag
        n = levels[-1].n
        P, R, Pf, Rf = self.get_prolongation(n)
        levels.append(multilevel_solver.level())

        if mg_option=='standard':
            # Form next level through Galerkin product
            Ag = R@A@P
            n=n//2
            Ag = Ag.astype(np.float64)  # convert from complex numbers, should have A.imag==0
            levels[-1].A = Ag.tocsr()
            levels[-1].Ag = Ag.tocsr()
            levels[-1].n = n

        elif mg_option=='learning':
            level_num = len(levels)
            model_key = f"level{level_num}"
            model_prob, model_value = models[model_key]
            model_prob.eval()
            model_value.eval()
            
            Ag = Rf @ A @ Pf
            A_uu = Ag[0:n*n,0:n*n]
            A_uv = Ag[0:n*n,n*n:]
            A_vu = Ag[n*n:,0:n*n]
            A_vv = Ag[n*n:,n*n:]

            stencil_uu = compute_stencil(A_uu,n,n,7)
            stencil_uv = compute_stencil(A_uv,n,n,5)
            stencil_vu = compute_stencil(A_vu,n,n,5)
            stencil_vv = compute_stencil(A_vv,n,n,7)
            stencil_test = [stencil_uu,stencil_uv,stencil_vu,stencil_vv]

            s_uu = torch.from_numpy(stencil_uu[n//2,n//2,:,:])
            s_uv = torch.from_numpy(stencil_uv[n//2,n//2,:,:])
            stencil = torch.zeros(1,6).double()
            stencil[0,0] = s_uu[0,1]
            stencil[0,1] = s_uu[0,3]
            stencil[0,2] = s_uu[1,1]
            stencil[0,3] = s_uu[1,2]
            stencil[0,4] = s_uu[1,3]
            stencil[0,5] = s_uv[0,1]

            prob = model_prob(stencil).squeeze()
            value = model_value(stencil).squeeze()
            stencils, s, ss = self.sparsify(prob,value,int(len(prob)*0.5),stencil_train)
            s_uu,s_uv,s_vu,s_vv = s
            ss_uu,ss_uv,ss_vu,ss_vv = ss
            stencil_uu,stencil_uv,stencil_vu,stencil_vv = stencils

            A_uu = compute_A_numpy(stencil_uu,stencil_uu.shape[0],stencil_uu.shape[1],stencil_uu.shape[2])
            A_uv = compute_A_numpy(stencil_uv,stencil_uv.shape[0],stencil_uv.shape[1],stencil_uv.shape[2])
            A_vu = compute_A_numpy(stencil_vu,stencil_vu.shape[0],stencil_vu.shape[1],stencil_vu.shape[2])
            A_vv = compute_A_numpy(stencil_vv,stencil_uu.shape[0],stencil_uu.shape[1],stencil_uu.shape[2])
            A_uu_uv = sp.sparse.hstack((A_uu,A_uv))
            A_vu_vv = sp.sparse.hstack((A_vu,A_vv))
            A_c = sp.sparse.vstack((A_uu_uv,A_vu_vv)).tocsr()
            A_c = A_c[0:n*n*2:2,:]
            A_c = A_c[:,0:n*n*2:2]

            n=n//2 
            levels[-1].A = A_c
            levels[-1].n=n
    
    def sparsify(self, prob, value, K, stencil_train):
        # prob = torch.sigmoid(X[0:8])
        # prob = X[0:8]
        prob = top_k(prob, K).squeeze()
        stencil = prob * value
        # stencil = value
        stencil = stencil.view(-1,1)
        stencil_uu = torch.from_numpy(stencil_train[0]).clone()
        stencil_uv = torch.from_numpy(stencil_train[1]).clone()
        stencil_vu = torch.from_numpy(stencil_train[2]).clone()
        stencil_vv = torch.from_numpy(stencil_train[3]).clone()
        m,n,_,_ = stencil_uu.shape
        s_uu = torch.zeros(5,5).double()
        ss_uu = torch.zeros(7,7).double()
        s_uv = torch.zeros(4,4).double()
        ss_uv = torch.zeros(5,5).double()

        s_vu = torch.zeros(4,4).double()
        ss_vu = torch.zeros(5,5).double()

        s_vv = torch.zeros(5,5).double()
        ss_vv = torch.zeros(7,7).double()

        s_uu[0,0] = 0
        s_uu[0,1] = stencil[0]
        s_uu[0,2] = 0
        s_uu[0,3] = stencil[1]
        s_uu[0,4] = 0
        s_uu[1,0] = stencil[1]
        s_uu[1,1] = stencil[2]
        s_uu[1,2] = stencil[3]
        s_uu[1,3] = stencil[4]
        s_uu[1,4] = stencil[0]
        s_uu[2,0] = 0
        s_uu[2,1] = stencil[3]
        s_uu[2,2] = -stencil[0]*4-stencil[1]*4-stencil[2]*2-4*stencil[3]-2*stencil[4]
        s_uu[2,3] = stencil[3]
        s_uu[2,4] = 0
        s_uu[3,0] = stencil[0]
        s_uu[3,1] = stencil[4]
        s_uu[3,2] = stencil[3]
        s_uu[3,3] = stencil[2]
        s_uu[3,4] = stencil[1]
        s_uu[4,0] = 0
        s_uu[4,1] = stencil[1]
        s_uu[4,2] = 0
        s_uu[4,3] = stencil[0]
        s_uu[4,4] = 0
        ss_uu[1:6,1:6] = s_uu

        s_vv[0,0] = 0
        s_vv[0,1] = stencil[1]
        s_vv[0,2] = 0
        s_vv[0,3] = stencil[0]
        s_vv[0,4] = 0
        s_vv[1,0] = stencil[0]
        s_vv[1,1] = stencil[2]
        s_vv[1,2] = stencil[3]
        s_vv[1,3] = stencil[4]
        s_vv[1,4] = stencil[1]
        s_vv[2,0] = 0
        s_vv[2,1] = stencil[3]
        s_vv[2,2] = -stencil[0]*4-stencil[1]*4-stencil[2]*2-4*stencil[3]-2*stencil[4]
        s_vv[2,3] = stencil[3]
        s_vv[2,4] = 0
        s_vv[3,0] = stencil[1]
        s_vv[3,1] = stencil[4]
        s_vv[3,2] = stencil[3]
        s_vv[3,3] = stencil[2]
        s_vv[3,4] = stencil[0]
        s_vv[4,0] = 0
        s_vv[4,1] = stencil[0]
        s_vv[4,2] = 0
        s_vv[4,3] = stencil[1]
        s_vv[4,4] = 0
        
        ss_vv[1:6,1:6] = s_vv

        s_uv[0,0] = 0
        s_uv[0,1] = stencil[5]
        s_uv[0,2] = stencil[5]
        s_uv[0,3] = 0
        s_uv[1,0] = -stencil[5]
        # s_uv[1,1] = stencil[6]
        # s_uv[1,2] = stencil[7]
        s_uv[1,3] = -stencil[5]
        s_uv[2,0] = -stencil[5]
        # s_uv[2,1] = stencil[8]
        # s_uv[2,2] = stencil[9]
        s_uv[2,3] = -stencil[5]
        s_uv[3,0] = 0
        s_uv[3,1] = stencil[5]
        s_uv[3,2] = stencil[5]
        s_uv[3,3] = 0
        ss_uv[0:4,0:4] = s_uv

        s_vu[0,0] = 0
        s_vu[0,1] = stencil[5]
        s_vu[0,2] = stencil[5]
        s_vu[0,3] = 0
        s_vu[1,0] = -stencil[5]
        # s_vu[1,1] = stencil[10]
        # s_vu[1,2] = stencil[11]
        s_vu[1,3] = -stencil[5]
        s_vu[2,0] = -stencil[5]
        # s_vu[2,1] = stencil[12]
        # s_vu[2,2] = stencil[13]
        s_vu[2,3] = -stencil[5]
        s_vu[3,0] = 0
        s_vu[3,1] = stencil[5]
        s_vu[3,2] = stencil[5]
        s_vu[3,3] = 0
        ss_vu[0:4,0:4] = s_vu

        ss_uu,_,_ = reorder_T(ss_uu.detach(),1)
        ss_uv,_,_ = reorder_T(ss_uv.detach(),2)
        ss_vu,_,_ = reorder_T(ss_vu.detach(),2)
        ss_vv,_,_ = reorder_T(ss_vv.detach(),1)

        s = (s_uu,s_uv,s_vu,s_vv)
        ss = (ss_uu,ss_uv,ss_vu,ss_vv)
        for i in range(2,m-2):
            for j in range(2,n-2):
                stencil_uu[i,j,:,:] = ss_uu
                stencil_uv[i,j,:,:] = ss_uv
                stencil_vu[i,j,:,:] = ss_vu
                stencil_vv[i,j,:,:] = ss_vv

        stencils = (stencil_uu,stencil_uv,stencil_vu,stencil_vv)
        return stencils, s, ss

