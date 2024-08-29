import torch
from .utils import *


small = 1e-30

class Functional():
    # Abstract class
    def __init__(self, domain) -> None:
        self.domain = domain
        x = domain.reshape(-1, domain.size(-1))
        self.c = costmatrix(x, x)
        self.x = x
    
    def forward(self, a):
        return None
    
    def __add__(self, G):
        assert G.domain.shape == self.domain.shape and torch.allclose(G.domain, self.domain)
        H = Functional(self.domain)
        H.forward = lambda a: self.forward(a) + G.forward(a)
        H.grad = lambda a: self.grad(a) + G.grad(a)
        return H
    
    def __rmul__(self, scalar):
        H = self.copy()
        H.forward = lambda a: scalar*self.forward(a)
        H.grad = lambda a: scalar*self.grad(a)
        return H
    
    def copy(self):
        H = Functional(self.domain)
        return H

    def __call__(self, a):
        return self.forward(a)

class PotentialEnergy(Functional):
    def __init__(self, domain, potential):
        super().__init__(domain)
        self.potential = potential(self.x)
    
    def forward(self, a):
        return a @ self.potential

    def grad(self, a):
        return self.potential
    
    def copy(self):
        H = PotentialEnergy(self.domain, lambda x: None)
        H.potential = self.potential.clone()
        return H

class Entropy(Functional):
    def __init__(self, domain, reference_measure=1):
        super().__init__(domain)
        self.ref = reference_measure
    
    def forward(self, a):
        return a @ torch.log(small + a/self.ref)

    def grad(self, a):
        return torch.log(small + a/self.ref) + 1

