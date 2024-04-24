import torch


class GradientDescentOptimizer:
    def __init__(self, model, w):
        self.model = model
        self.w = None
    
    def step(self, alpha, beta, w):
        if self.w == None:
            self.old_w = self.model.w.clone()

        old_w = self.model.w.clone()
        
        current_w = self.w - alpha * self.grad * self.w + beta * (self.w - old_w)
        return current_w
        

        