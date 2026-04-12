import sys
import os

# Thêm thư mục src vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tinytorch.tensor import Tensor
import numpy as np
import math


class Sigmoid:
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(1 / (1 + np.exp(-(x.data))))
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    
    
class ReLU:
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.maximum(0,x.data))
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    
class Tanh:
    def tanh1(self, x):
        return -(1 - np.exp(2 * x))/(1 + np.exp(2 * x))
    
    def tanh2(self, x):
        return (1 - np.exp(-(2 * x)))/(1 + np.exp(-(2 * x)))
    
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.where(x.data > 0,self.tanh2(x.data),self.tanh1(x.data)))
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
class GELU:
    def forward(self, x: Tensor) -> Tensor:
        erf = np.vectorize(math.erf)
        return Tensor(1/2 * x.data * (1 + erf(x.data/math.sqrt(2))))
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    
class Softmax:
    def forward(self, x: Tensor,dim: int = -1) -> Tensor: 
        max_x = np.max(x.data, axis=dim, keepdims=True)
        exp_sum = np.sum(np.exp(x.data - max_x),axis=dim, keepdims=True) 
        return Tensor(np.exp(x.data - max_x) / exp_sum)   
    
    def __call__(self, x: Tensor,dim: int = -1) -> Tensor:
        return self.forward(x) 