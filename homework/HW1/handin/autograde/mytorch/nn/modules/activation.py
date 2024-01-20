import numpy as np


class Identity:
    
    def forward(self, Z):
    
        self.A = Z
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.ones(self.A.shape, dtype="f")
        
        return dAdZ


class Sigmoid:
    
    def forward(self, Z):
    
        self.A = # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = # TODO
        
        return dAdZ


class Tanh:
    
    def forward(self, Z):
    
        self.A = # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = # TODO
        
        return dAdZ


class ReLU:
    
    def forward(self, Z):
    
        self.A = # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = # TODO
        
        return dAdZ
        
        
