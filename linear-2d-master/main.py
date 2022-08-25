import torch
import matplotlib.pyplot as plt
import csv

x_array, y_array = [None] * 1001, [None] * 1001

with open('length_weight.csv', mode="r") as csv_file:
    reader = csv.reader(csv_file)

    i = 0

    for item in reader:
        x_array[i] = item[0]
        y_array[i] = item[1]
        i += 1

# Observed/training input and output
# x_train = [[1], [1.5], [2], [3], [4], [5], [6]]
x_train = torch.tensor(x_array).reshape(-1, 1)
# y_train = [[5], [3.5], [3], [4], [3], [1.5], [2]]
y_train = torch.tensor(y_array).reshape(-1, 1)


class LinearRegressionModel:

    def __init__(self):
        # Model variables
        # requires_grad enables calculation of gradients
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

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
for epoch in range(1000):
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
