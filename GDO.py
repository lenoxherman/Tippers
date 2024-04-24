import torch

class GradientDescentOptimizer:
    def __init__(self, learning_rate, w):
        self.learning_rate = learning_rate
        self.w = None

    def optimize(self, model, loss, X, y):
        if self.w == None:
            self.w = torch.rand((X.size()[1]))
        # Compute the gradient of the loss with respect to the model parameters
        gradient = model.gradient(loss, X, y)
        # Update the model parameters
        model.update(gradient, self.learning_rate)
    
    def step(self, model, loss, X, y):
        self.optimize(model, loss, X, y)
        