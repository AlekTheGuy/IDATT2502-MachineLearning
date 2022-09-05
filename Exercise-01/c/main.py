import torch
import matplotlib.pyplot as plt
import pandas as panda

data = panda.read_csv("day_head_circumference.csv")
y_train = data.pop("head circumference")
x_train = torch.tensor(data.to_numpy(), dtype=torch.float).reshape(-1, 1)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float).reshape(-1, 1)


class LinearRegressionModel:

    def __init__(self):
        # Model variables'
        # requires_grad enables calculation of gradients
        self.W = torch.tensor([[0.0]], dtype=torch.float, requires_grad=True)
        self.b = torch.tensor([[0.0]], dtype=torch.float, requires_grad=True)

    # Predictor
    def f(self, x):

        # @ corresponds to matrix multiplication
        return 20 * torch.sigmoid((x @ self.W + self.b)) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.000001)
for epoch in range(100000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s    " %
      (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.figure('Nonlinear regression 2d')
plt.title('Predict head circumference based on age')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_train, y_train)
x = torch.arange(torch.min(x_train), torch.max(x_train), 1.0).reshape(-1, 1)
y = model.f(x).detach()
plt.plot(x, y, color='orange',
         label='$f(x) = 20\sigma(xW + b) + 31$ \n$\sigma(z) = \dfrac{1}{1+e^{-z}}$')

plt.legend()
plt.show()
