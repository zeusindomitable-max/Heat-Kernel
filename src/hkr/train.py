import torch
from tqdm import trange

class HKRTrainer:
    """
    τ-evolution minimization of HKR functional.
    """

    def __init__(self, functional, lr=1e-2, steps=100):
        self.F = functional
        self.lr = lr
        self.steps = steps

    def run(self, field):
        f = field.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([f], lr=self.lr)

        for _ in trange(self.steps, desc="HKR evolution"):
            optimizer.zero_grad()
            energy = self.F.energy(f)
            energy.backward()
            optimizer.step()
        return f.detach()
