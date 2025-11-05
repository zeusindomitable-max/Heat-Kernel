import torch
from hkr.geometry import S2Manifold
from hkr.heatkernel import HeatKernel
from hkr.variational import HKRFunctional

def test_hkr_energy():
    M = S2Manifold(resolution=16)
    HK = HeatKernel(M)
    F = HKRFunctional(M, HK)
    field = M.random_field()
    E = F.energy(field)
    assert torch.isfinite(E)
