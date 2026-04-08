import numpy as np

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion


class Tensor:
    def __init__(self,data):
        self.data = np.array(data, dtype = np.float32)
        
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        
    # override Python magic method __add__()    
    def __add__(self, other):
        
        #Apply NumPy’s broadcasting rules automatically
        if isinstance(other,Tensor):
            return Tensor(self.data + other.data)
        
        #Scalar broadcasting
        else:
            return Tensor(self.data + other)
        
    def __sub__(self, other):
         if isinstance(other,Tensor):
             return Tensor(self.data - other.data)
         
         else:
             return Tensor(self.data - other)      
         
    def __mul__(self, other):
        
        # dot product for 2 matrix
        if isinstance(other,Tensor):
             return Tensor(self.data * other.data)
         
        else:
             return Tensor(self.data * other)  
    
    
    def __truediv__(self, other):
        if isinstance(other,Tensor):
             return Tensor(self.data / other.data)
         
        else:
             return Tensor(self.data / other)  
    
    def matmul(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"Matrix multiplication require Tensor.\n Got {type(other).__name__} ")
        
        if other.size == 1 or self.size == 1:
            raise ValueError("Matrix multiplication require at lease 1D Tensor\n"
                             f"Got {self.shape} @ {other.shape}\n")
            
        if self.shape[-1] != other.shape[-2]:
            raise ValueError(f"Failed matrix mutiplication rule\n. Got {self.shape} @ {other.shape}")    
        
        
        a = self.data
        b = other.data
        
        
        if len(self.shape) == 2 and len(other.shape) == 2:
            result = np.zeros(shape = (a.shape[0],b.shape[1]))
            
            for i in range(self.shape[0]):
                for j in range(other.shape[1]):
                    result[i][j] = np.dot(a[i,:], b[:,j])
                
                
            return Tensor(result)  
        
        
        else:
            return Tensor(np.matmul(a,b))  
      
    
    def reshape(self, *shape):           
        if len(shape) == 1 and isinstance(shape, (tuple,list)):
            axes = np.array(shape[0])
        else:
            axes = np.array(shape)    
        
        
        loc = np.where (axes == -1)
        
        if len(loc[0]) >= 2:
            raise ValueError("can only specific one unknown dimension.")
        elif len(loc[0]) == 1:
            axes[loc[0][0]] = self.size // (np.prod(axes) * -1)
            
        if np.prod(axes) != self.size:
            raise ValueError(
                'Total elements don\'t match\n'
                f'Original tensor size {self.size}\n'
                f'Input shape size {np.prod(axes)}\n' 
            )  
         
        new_tensor =  Tensor(np.reshape(self.data,tuple(axes))) 
                       
        return new_tensor
    
    
    def transpose(self, dim0 = None, dim1 = None):
        axes = list(range(len(self.shape)))
        
        if dim0 == None and dim1 == None:
            axes[-2], axes[-1] = axes[-1], axes[-2]
        else:
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            
        return Tensor(np.transpose(self.data,axes = axes))         
                
    def sum(self, axis = None, keepdims = False):
        return np.sum(self.data, axis = axis, keepdims = keepdims)
    
    def mean(self, axis = None, keepdims = False):
        return np.mean(self.data, axis = axis, keepdims = keepdims)
    
    def max(self, axis = None, keepdims = False):
        return np.max(self.data, axis = axis, keepdims = keepdims)            
    
