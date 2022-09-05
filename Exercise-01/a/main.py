import torch
import matplotlib.pyplot as plt
import pandas as panda

data = panda.read_csv("length_weight.csv")
x_train = data.pop("length")
x_train = torch.tensor(x_train.to_numpy(), dtype=torch.double).reshape(-1, 1)
y_train = torch.tensor(data.to_numpy(), dtype=torch.double).reshape(-1, 1)


class LinearRegressionModel:

    def __init__(self):
        # Model variables
        # requires_grad enables calculation of gradients
        self.W = torch.tensor([[0.0]], dtype=torch.double, requires_grad=True)
        self.b = torch.tensor([[0.0]], dtype=torch.double, requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.01)
for epoch in range(300000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" %
      (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
# x = [[1], [6]]]
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()
