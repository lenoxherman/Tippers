import torch


class GradientDescentOptimizer:
    def __init__(self, model, w):
        self.model = model
        self.w = None
    
    def step(self, alpha, beta, X, y, w):
        #taken from Daniela's logistic regression
        if self.w == None:
            self.old_w = self.model.w.clone()
        
        current_w = self.model.w.clone()
        self.model.w = self.model.w - self.model.grad(X, y).flatten() * alpha + beta * (self.model.w - self.old_w)
        self.old_w = current_w
        

        