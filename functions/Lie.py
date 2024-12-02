import torch
from scipy.spatial.transform import Rotation

basis = torch.tensor([
    [[0.,0.,0.],[0.,0.,-1.],[0.,1.,0.]],
    [[0.,0.,1.],[0.,0.,0.],[-1.,0.,0.]],
    [[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]]])

def hat(v): # R^3 -> so(3)
    return torch.einsum('...i, ijk -> ...jk', v, basis)

def vee(A): # so(3) -> R^3
    return torch.stack([A[...,2,1], A[...,0,2], A[...,1,0]], dim=-1)

def logSO3(R): # SO(3) -> so(3)
    return hat(Logmap(R))

def expso3(A): # so(3) -> SO(3)
    return torch.linalg.matrix_exp(A)

def Logmap(R): # SO(3) -> R^3
    return torch.tensor(Rotation.from_matrix(R.detach().numpy()).as_rotvec(), dtype=torch.float32)

def Expmap(v): # R^3 -> SO(3)
    return expso3(hat(v))

def Omega(R): # SO(3) -> R
    return torch.arccos((torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1) / 2)

# R = torch.tensor(Rotation.random(5).as_matrix(), dtype=torch.float32)
# A = logSO3(R)
# v = Logmap(R)
# assert torch.allclose(R, expso3(A), atol=1e-6)
# assert torch.allclose(R, Expmap(v), atol=1e-6)