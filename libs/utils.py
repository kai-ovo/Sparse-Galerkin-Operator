import numpy as np
import torch
import scipy as sp
import pyamg
import random
from scipy.sparse import csr_matrix

def diffusion_stencil_2d(epsilon=1.0, theta=0.0, type='FE'):
    eps = float(epsilon)  # for brevity
    theta = float(theta)

    C = np.cos(theta)
    S = np.sin(theta)
    CS = C*S
    CC = C**2
    SS = S**2

    if type == 'FE' :
        a = (-1*eps - 1)*CC + (-1*eps - 1)*SS + (3*eps - 3)*CS
        b = (2*eps - 4)*CC + (-4*eps + 2)*SS
        c = (-1*eps - 1)*CC + (-1*eps - 1)*SS + (-3*eps + 3)*CS
        d = (-4*eps + 2)*CC + (2*eps - 4)*SS
        e = (8*eps + 8)*CC + (8*eps + 8)*SS

        stencil = np.array([[a, b, c],
                            [d, e, d],
                            [c, b, a]]) / 6.0

    elif type == 'FD':

        a = -0.5*(eps - 1)*CS
        b = -(eps*SS + CC)
        c = -a
        d = -(eps*CC + SS)
        e = 2.0*(eps + 1)

        stencil = np.array([[a+c, d-2*c, 2*c],
                            [b-2*c, e+4*c, b-2*c],
                            [2*c, d-2*c, a+c]])

    return stencil

def elasticity(E=1e-5,nu=0.3,by=1):
    mu = E*(1+2*nu)
    lam = E*nu/((1+nu)*(1-2*nu))
    k17 = (-lam*by*by-2*mu*by*by+lam+mu)/(4*(2*lam*by*by+lam+4*(by*by+1)*mu))
    k14 = -(2*lam*by*by+4*mu*by*by+lam+mu)/(2*(2*lam*by*by+lam+4*(by*by+1)*mu))
    k11 = k17
    k12 = ((by**2-1)*lam+2*(by**2-2)*mu)/(2*(2*lam*by*by+lam+4*(by**2+1)*mu))
    k18 = k12
    k15 = 1
    k19 = k17
    k13 = k17
    k16 = k14
    s1 = np.array([[k17,k18,k19],[k14,k15,k16],[k11,k12,k13]])
    k = (3*by*(lam+mu))/(8*(2*lam*by*by+lam+4*(by*by+1)*mu))
    s2 = np.array([[k,0,-k],[0,0,0],[-k,0,k]])
    return s1,s2

def map_2_to_1(grid_size_x=8,grid_size_y=8,stencil_size=3):
    # maps 2D coordinates to the corresponding 1D coordinate in the matrix.
    k = np.zeros((grid_size_x, grid_size_y, stencil_size, stencil_size))
    M = np.reshape(np.arange(grid_size_x * grid_size_y), (grid_size_x, grid_size_y))
    M = np.concatenate([M, M], 0)
    M = np.concatenate([M, M], 1)
    for i in range(stencil_size):
        I = (i - stencil_size//2) % grid_size_x
        for j in range(stencil_size):
            J = (j - stencil_size//2) % grid_size_y
            k[:, :, i, j] = M[I:I + grid_size_x, J:J + grid_size_y]
    return k

def transpose_stencil_numpy(s):
    a = s[0,0]
    b = s[0,1]
    c = s[0,2]
    d = s[1,0]
    e = s[1,1]
    f = s[1,2]
    g = s[2,0]
    h = s[2,1]
    i = s[2,2]
    return np.array([[i,h,g],[f,e,d],[c,b,a]])

def reorder(X,option):
    new_IDX = []
    old_IDX = []
    if len(X.shape)<3:
        if option == 1:
            m,n = X.shape
            Y = torch.zeros(m,n).double()
            IDX = []
            for i in range(m):
                for j in range(n):
                    if (i%2)==(j%2):
                        IDX.append((i,j))
            for (i,j) in IDX:
                new_i = (i+j)//2
                new_j = (-i+j)//2+m//2
                Y[new_i,new_j] = X[i,j]
                new_IDX.append(new_i*n+new_j)        
        else:
            m,n = X.shape
            Y = torch.zeros(m,n).double()
            IDX = []
            for i in range(m):
                for j in range(n):
                    if (i%2)!=(j%2):
                        IDX.append((i,j))
            for (i,j) in IDX:
                new_i = (i+j)//2
                new_j = (-i+j)//2+m//2
                Y[new_i,new_j] = X[i,j]
                new_IDX.append(new_i*n+new_j)
    elif len(X.shape)==3:
        z,m,n = X.shape
        Y = torch.zeros(z,m,n).double()
        for i in range(z):
            Y[i,:,:] = reorder(X[i,:,:],1)[0]
    return Y,new_IDX,old_IDX
    
def reorder_T(X,option):
    new_IDX=[]
    old_IDX = []
    if option == 1:
        m,n = X.shape
        Y = torch.zeros(m,n).double()
        IDX = []
        for i in range(m):
            if i<=m//2:
                for j in range(n//2-i,n//2+i+1):
                    IDX.append((i,j))
            else:
                for j in range(n//2-m+i+1,n//2+m-i):
                    IDX.append((i,j))
        for (i,j) in IDX:
            new_i = (i-j)+m//2
            new_j = (i+j)-m//2
            Y[new_i,new_j] = X[i,j]
            new_IDX.append(new_i*n+new_j)
            old_IDX.append(i*n+j)
    elif option == 2:
        m,n = X.shape
        Y = torch.zeros(m,n).double()
        IDX = []
        for i in range(m):
            if i<m//2:
                for j in range(n//2-i-1,n//2+i+1):
                    IDX.append((i,j))
            elif i<(m//2)*2:
                for j in range(n//2-m+i+1,n//2+m-i-1):
                    IDX.append((i,j))

        for (i,j) in IDX:
            if 0<=i<m-1 and 0<=j<n-1:
                new_i = (i-j)+m//2
                new_j = (i+j)-m//2+1
                # print((i,j),(new_i,new_j))
                Y[new_i,new_j] = X[i,j]
                new_IDX.append(new_i*n+new_j)
                old_IDX.append(i*n+j)
    return Y,new_IDX,old_IDX

def map_2_to_1_old(grid_size=8):
    # maps 2D coordinates to the corresponding 1D coordinate in the matrix.
    k = np.zeros((grid_size, grid_size, 3, 3))
    M = np.reshape(np.arange(grid_size ** 2), (grid_size, grid_size))
    M = np.concatenate([M, M], 0)
    M = np.concatenate([M, M], 1)
    for i in range(3):
        I = (i - 1) % grid_size
        for j in range(3):
            J = (j - 1) % grid_size
            k[:, :, i, j] = M[I:I + grid_size, J:J + grid_size]
    return k

def map_2_to_1_numpy(grid_size=8):
    # maps 2D coordinates to the corresponding 1D coordinate in the matrix.
    k = np.zeros((grid_size, grid_size, 3, 3))
    M = np.reshape(np.arange(grid_size ** 2), (grid_size, grid_size)).T
    M = np.concatenate([M, M], 0)
    M = np.concatenate([M, M], 1)
    for i in range(3):
        I = (i - 1) % grid_size
        for j in range(3):
            J = (j - 1) % grid_size
            k[:, :, i, j] = M[I:I + grid_size, J:J + grid_size]
    return k

def map_2_to_1_torch(grid_size=8):
    # maps 2D coordinates to the corresponding 1D coordinate in the matrix.
    k = np.zeros((grid_size, grid_size, 3, 3))
    M = np.reshape(np.arange(grid_size ** 2), (grid_size, grid_size)).T
    M = np.concatenate([M, M], 0)
    M = np.concatenate([M, M], 1)
    for i in range(3):
        I = (i - 1) % grid_size
        for j in range(3):
            J = (j - 1) % grid_size
            k[:, :, i, j] = M[I:I + grid_size, J:J + grid_size]
    return k

def get_p_matrix_indices_one(grid_size):
    K = map_2_to_1(grid_size=grid_size)
    indices = []
    for ic in range(grid_size // 2):
        i = 2 * ic + 1
        for jc in range(grid_size // 2):
            j = 2 * jc + 1
            J = int(grid_size // 2 * jc + ic)
            for k in range(3):
                for m in range(3):
                    I = int(K[i, j, k, m])
                    indices.append([I, J])

    return np.array(indices)

def compute_stencil(A, grid_size_x,grid_size_y,stencil_size):
    indices = get_indices_compute_A_one(grid_size_x,grid_size_y,stencil_size)
    stencil = np.array(A[indices[:, 0], indices[:, 1]]).reshape((grid_size_x, grid_size_y, stencil_size, stencil_size))
    return stencil

def compute_stencil_numpy(A, grid_size):
    indices = get_indices_compute_A_one_numpy(grid_size)
    stencil = np.array(A[indices[:, 0], indices[:, 1]]).reshape((grid_size, grid_size, 3, 3))
    return stencil

def get_indices_compute_A_one(grid_size_x=8,grid_size_y=8,stencil_size=3):
    indices = []
    K = map_2_to_1(grid_size_x,grid_size_y,stencil_size)
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            I = int(K[i, j, stencil_size//2, stencil_size//2])
            for k in range(stencil_size):
                for m in range(stencil_size):
                    J = int(K[i, j, k, m])
                    indices.append([I, J])

    return np.array(indices)

def get_indices_compute_A_one_numpy(grid_size):
    indices = []
    K = map_2_to_1_numpy(grid_size=grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            I = int(K[i, j, 1, 1])
            for k in range(3):
                for m in range(3):
                    J = int(K[i, j, k, m])
                    indices.append([I, J])

    return np.array(indices)

def compute_A_indices(grid_size_x=8,grid_size_y=8,stencil_size=3):
    K = map_2_to_1(grid_size_x,grid_size_y,stencil_size)
    A_idx = []
    stencil_idx = []
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            I = int(K[i, j, stencil_size//2, stencil_size//2])
            for k in range(stencil_size):
                for m in range(stencil_size):
                    J = int(K[i, j, k, m])
                    A_idx.append([I, J])
                    stencil_idx.append([i, j, k, m])
    return np.array(A_idx), stencil_idx

def compute_A_indices_torch(grid_size):
    K = map_2_to_1_torch(grid_size=grid_size)
    A_idx = []
    stencil_idx = []
    for i in range(grid_size):
        for j in range(grid_size):
            I = int(K[i, j, 1, 1])
            for k in range(3):
                for m in range(3):
                    J = int(K[i, j, k, m])
                    A_idx.append([I, J])
                    stencil_idx.append([i, j, k, m])
    return np.array(A_idx), stencil_idx

def compute_p2(P_stencil, grid_size):
    indexes = get_p_matrix_indices_one(grid_size)
    P = sp.sparse.csr_matrix(arg1=(P_stencil.reshape(-1), (indexes[:, 1], indexes[:, 0])),
                   shape=((grid_size//2) ** 2, (grid_size) ** 2))

    return P

def compute_p2_old(P_stencil, grid_size):
    indexes = get_p_matrix_indices_one_old(grid_size)
    P = sp.sparse.csr_matrix(arg1=(P_stencil.reshape(-1), (indexes[:, 1], indexes[:, 0])),
                             shape=((grid_size//2) ** 2, (grid_size) ** 2))

    return P

def compute_p2_numpy(P_stencil, grid_size):
    indexes = get_p_matrix_indices_one_numpy(grid_size)
    P = sp.sparse.csr_matrix(arg1=(P_stencil.reshape(-1), (indexes[:, 1], indexes[:, 0])),
                   shape=((grid_size//2) ** 2, (grid_size) ** 2))

    return P

def get_p_matrix_indices_one_old(grid_size):
    K = map_2_to_1_old(grid_size=grid_size)
    indices = []
    for ic in range(grid_size // 2):
        i = 2 * ic + 1
        for jc in range(grid_size // 2):
            j = 2 * jc + 1
            J = int(grid_size // 2 * jc + ic)
            for k in range(3):
                for m in range(3):
                    I = int(K[i, j, k, m])
                    indices.append([I, J])

    return np.array(indices)

def get_p_matrix_indices_one_numpy(grid_size):
    K = map_2_to_1_numpy(grid_size=grid_size)
    indices = []
    for ic in range(grid_size // 2):
        i = 2 * ic + 1
        for jc in range(grid_size // 2):
            j = 2 * jc + 1
            J = int(grid_size // 2 * jc + ic)
            for k in range(3):
                for m in range(3):
                    I = int(K[i, j, k, m])
                    indices.append([I, J])

    return np.array(indices)

def prolongation_fn(grid_size):
#     grid_size = int(math.sqrt(A.shape[0]))
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
    return compute_p2_old(P_stencils, grid_size).astype(np.double)  # imaginary part should be zero

def compute_A(P_stencil, grid_size_x,grid_size_y,stencil_size):
    A,_ = compute_A_indices(grid_size_x,grid_size_y,stencil_size)
    P = torch.sparse.DoubleTensor(torch.LongTensor(A.T), P_stencil.view(-1), (grid_size_x*grid_size_y,grid_size_x*grid_size_y))
    return P

def compute_A_numpy(P_stencil, grid_size_x,grid_size_y,stencil_size):
    A,_ = compute_A_indices(grid_size_x,grid_size_y,stencil_size)
    P = csr_matrix(arg1=(P_stencil.reshape((-1)), (A[:, 0], A[:, 1])),
                   shape=(grid_size_x*grid_size_y,grid_size_x*grid_size_y))
    return P
def compute_A_numpy2(stencils,grid_size):
    A,_ = compute_A_indices_numpy(grid_size)
    P_numpy = sp.sparse.csr_matrix(arg1=(stencils.reshape(-1), (A[:, 0], A[:, 1])),
               shape=(grid_size ** 2, grid_size  ** 2))
    return P_numpy

def compute_A_indices_numpy(grid_size):
    K = map_2_to_1_numpy(grid_size=grid_size)
    A_idx = []
    stencil_idx = []
    for i in range(grid_size):
        for j in range(grid_size):
            I = int(K[i, j, 1, 1])
            for k in range(3):
                for m in range(3):
                    J = int(K[i, j, k, m])
                    A_idx.append([I, J])
                    stencil_idx.append([i, j, k, m])
    return np.array(A_idx), stencil_idx


def compute_A_sparse(stencils,grid_size):
    A,_ = compute_A_indices_numpy(grid_size)
    P_numpy = sp.sparse.csr_matrix(arg1=(stencils.reshape(-1), (A[:, 0], A[:, 1])),
            shape=(grid_size ** 2, grid_size  ** 2))
    return P_numpy

def compute_A_torch(P_stencil, grid_size):
    A,_ = compute_A_indices_torch(grid_size)
    P = torch.sparse.DoubleTensor(torch.LongTensor(A.T).to(P_stencil.get_device()), 
                                  P_stencil.view(-1), (grid_size**2,grid_size**2))
    return P

def res_matrix(n):
    res_stencil = torch.zeros(3,3).double()#.to(device)
    k=4
    res_stencil[0,1] = 1/k
    res_stencil[1,1] = 2/k
    res_stencil[2,1] = 1/k
    R = pyamg.gallery.stencil_grid(res_stencil.cpu(), (n,n)).tocsr()
    idx=[]
    for i in range(n//2):
        idx=idx+list(range((i+1)*2*n-n,(i+1)*2*n))
    #R = torch.Tensor(R.toarray()).double()
    R = R[idx]
    return R

def coo_to_tensor(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    shape = coo.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape),requires_grad = False)

def top_k(logits,k,softmax_on):
    if softmax_on:
        y_soft = logits.softmax(-1)
    else:
        y_soft = logits
#     index = torch.topk(torch.abs(y_soft),k)[1]
    index = torch.topk(y_soft,k)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret

def topk_dim(logits,k,softmax_on,dim=-1):
    if softmax_on:
        y_soft = logits.softmax(-1)
    else:
        y_soft = logits
    index = torch.topk(y_soft,k,dim=dim)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret
    
def set_global_seed(seed: int) -> None:
    """
    Sets random seed into PyTorch, Numpy and Random.
    Args:
        seed: random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def update_params(param_dict : dict, **params):
    for key, value in params.items():
        param_dict[key] = value
