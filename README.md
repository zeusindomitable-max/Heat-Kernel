# HKR-MVP: Heat Kernel Regularization Framework

A minimal yet mathematically rigorous implementation of the **Heat Kernel Regularization (HKR)** framework for variational field analysis, spectral geometry, and physical regularization of PDE systems. This MVP is structured for clarity, reproducibility, and direct extension into differentiable optimization workflows (e.g., via PyTorch).

---

## 🔍 Overview

HKR (Heat Kernel Regularization) provides a natural smoothing and variational regularization technique derived from the **asymptotic expansion of the heat kernel**:

$$
K(x, x'; \tau) = (4\pi\tau)^{-d/2} e^{-\frac{|x - x'|^2}{4\tau}} [I + \tau R(x) + \mathcal{O}(\tau^2)]
$$

The approach reformulates functional regularization and renormalization in geometric and spectral terms, offering strong analytical control in variational PDEs and machine-learning-inspired field models.

This repository contains a minimal working prototype implementing the key computational components:

* **geometry.py** – Core manifold and metric utilities (S², S²×S¹)
* **heatkernel.py** – Core HKR operator, asymptotic expansion, and spectral convolution
* **variational.py** – Variational energy formulation and gradient computation
* **train.py** – τ-evolution integrator for minimizing the HKR variational energy
* **tests/** – Minimal PyTest-based validation suite
* **examples/** – Demonstration notebooks for spherical domains

---

## 🧩 Project Structure

```
hkr-mvp/
│
├── src/
│   └── hkr/
│       ├── __init__.py
│       ├── geometry.py
│       ├── heatkernel.py
│       ├── variational.py
│       └── train.py
│
├── tests/
│   └── test_hkr.py
│
├── examples/
│   └── hkr_experiments.ipynb
│
├── README.md
├── setup.py
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/hkr-mvp.git
cd hkr-mvp
pip install -e .
```

This will install the HKR package in editable mode. You can then import modules as:

```python
from hkr.geometry import S2Manifold
from hkr.heatkernel import HeatKernel
```

---

## 🧠 Usage Example

```python
from hkr.geometry import S2Manifold
from hkr.heatkernel import HeatKernel
from hkr.variational import HKRFunctional

M = S2Manifold(resolution=64)
HK = HeatKernel(M)
F = HKRFunctional(M, HK)

energy, grad = F.energy_and_grad(field=M.random_field())
print("Initial energy:", energy.item())
```

---

## 🧪 Testing

```bash
pytest -v tests/
```

---

## 📖 References

* B. S. DeWitt, *Dynamical Theory of Groups and Fields*, 1965.
* S. Minakshisundaram & Å. Pleijel, *Some Properties of the Eigenfunctions of the Laplace Operator on Riemannian Manifolds*, 1949.
* R. Seeley, *Complex Powers of an Elliptic Operator*, 1967.
* Gilkey, *Invariance Theory, the Heat Equation, and the Atiyah–Singer Index Theorem*, 1984.

---

## 🧮 Citation

If you use HKR-MVP for research, please cite:

```
@software{HKR_MVP_2025,
  author = {Hari hardiyan},
  title = {Heat Kernel Regularization (HKR) MVP Framework},
  year = {2025},
  url = {https://github.com/zeusindomitable-max/Heat-Kernel},
  version = {v1.0.0},
  doi = {10.5281/zenodo.xxxxxxx}
}
```

---

# 💝 Support

Love this project? Help me keep building:

**ETH:** ` 0x7cc8686f434cf9b2f274f46fcf73ba6394635b48`

**BTC:** `1LUD9c2hYUERgPmtZCcUitDg8rgrNHfoYP`

**SOL:** `7mp34H3DEdBu5SxWtgkoM6QApYVwKyaY4P1Um7fcnMjZ`


Even small amounts help cover coffee ☕ and server costs!

## 🧭 Notes

This repository is the **core computational base** for HKR research and serves as a reproducible reference for heat kernel–based regularization schemes in mathematical physics and geometric learning.

For future extensions:

* Add PyTorch-based automatic differentiation.
* Implement spectral fitting for manifold Laplacians.
* Integrate numerical renormalization (RG) flows.

---

**Author:** Hari Hardiyan 
**License:** Dual License (Academic + Commercial) © 2025 Hari Hardiyan.
See LICENSE
 for full terms
