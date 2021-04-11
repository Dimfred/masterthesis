class Accumulator:
    def __init__(self, optim, accumulation_steps):
        pass

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
