import torch

class HKRFunctional:
    """
    Variational energy functional based on HKR.
    """

    def __init__(self, manifold, kernel):
        self.M = manifold
        self.HK = kernel

    def energy(self, field):
        """
        E[f] = ½ ⟨f, e^{τΔ} f⟩
        """
        return 0.5 * self.HK.kernel_norm(field)

    def grad(self, field):
        """
        Gradient: ∂E/∂f = e^{τΔ} f
        """
        return self.HK.apply(field)

    def energy_and_grad(self, field):
        E = self.energy(field)
        g = self.grad(field)
        return E, g
