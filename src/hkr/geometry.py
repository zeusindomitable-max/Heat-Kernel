import torch
import math

class S2Manifold:
    """
    Discretized 2-sphere (S²) geometry for HKR.
    Supports uniform sampling and metric computation.
    """

    def __init__(self, resolution: int = 64, device: str = "cpu"):
        self.res = resolution
        self.device = device
        self.theta, self.phi = self._generate_grid()
        self.metric_det = torch.sin(self.theta)
        self.area_element = 4 * math.pi / (self.res ** 2)

    def _generate_grid(self):
        t = torch.linspace(0, math.pi, self.res, device=self.device)
        p = torch.linspace(0, 2 * math.pi, self.res, device=self.device)
        theta, phi = torch.meshgrid(t, p, indexing="ij")
        return theta, phi

    def laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """
        Approximate Laplacian on S² using finite differences.
        """
        dtheta = math.pi / self.res
        dphi = 2 * math.pi / self.res
        sin_theta = torch.sin(self.theta)

        f_tt = (torch.roll(field, -1, 0) - 2 * field + torch.roll(field, 1, 0)) / (dtheta ** 2)
        f_pp = (torch.roll(field, -1, 1) - 2 * field + torch.roll(field, 1, 1)) / (dphi ** 2)

        term1 = f_tt
        term2 = (1 / (sin_theta ** 2 + 1e-9)) * f_pp
        term3 = torch.cos(self.theta) / (sin_theta + 1e-9) * (torch.roll(field, -1, 0) - torch.roll(field, 1, 0)) / (2 * dtheta)
        return term1 + term2 + term3

    def random_field(self, scale=0.1):
        return scale * torch.randn((self.res, self.res), dtype=torch.float64, device=self.device)
