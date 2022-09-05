import torch
import matplotlib.pyplot as plt
import pandas as panda
from mpl_toolkits.mplot3d import Axes3D
import collections

data = panda.read_csv("day_length_weight.csv", dtype='float')
y_train = data.pop("day")
x_train = torch.tensor(data.to_numpy(), dtype=torch.float).reshape(-1, 2)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float).reshape(-1, 1)


class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], dtype=torch.double, requires_grad=True).reshape(
            2, 1)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], dtype=torch.double, requires_grad=True)
        self.W = torch.rand((2, 1), dtype=torch.float, requires_grad=True)
        self.b = torch.rand((1, 1), dtype=torch.float, requires_grad=True)

    # Predictor

    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.00001)
for epoch in range(300000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" %
      (model.W, model.b, model.loss(x_train, y_train)))

x_axis = x_train.t()[0]
y_axis = x_train.t()[1]

fig = plt.figure("Task 2, linear regression in 3d")
ax = fig.add_subplot(projection='3d')
ax.scatter(x_axis.numpy(), y_axis.numpy(), y_train.numpy(),
           label='$(x^{(i)},y^{(i)}, z^{(i)})$')
ax.scatter(x_axis.numpy(), y_axis.numpy(), model.f(
    x_train).detach().numpy(), label='$\\hat y = f(x) = xW+b$', color="orange")
ax.legend()
plt.show()
