import numpy as np

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        error  = # TODO 
        L      = np.sum(error) / N
        
        return L
    
    def backward(self):
    
        dLdA = # TODO 
        
        return dLdA

class CrossEntropyLoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        Ones   = np.ones((C, 1), dtype="f")

        self.softmax = # TODO 
        crossentropy = # TODO 
        L = np.sum(crossentropy) / N
        
        return L
    
    def backward(self):
    
        dLdA = # TODO 
        
        return dLdA
