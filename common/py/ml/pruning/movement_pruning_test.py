import torch
import torch.nn as nn

from pruner import Movement

model = nn.Sequential(nn.Linear(2, 2))

print("original parameters")
for k, x in model.named_parameters():
    print(k, x)

pruner = Movement(model)

print("original and scores parameters")
for k, x in model.named_parameters():
    print(k, x)

print("0 masked weight and bias")
print(model[0].weight)
print(model[0].bias)
print("50 masked weight and bias")
pruner.prune(0.5)
print(model[0].weight)
print(model[0].bias)

opt = torch.optim.SGD(model.parameters(), lr=0.1)

x = torch.tensor([[1., 2.]])
y = model(x)
print("y")
print(y)
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(y, torch.tensor([1]))
loss.backward()

print("grad of weight")
print(model[0].parametrizations.weight.original.grad)
print("grad of weight score")
print(model[0].parametrizations.weight[0].scores.grad)
print("grad of bias")
print(model[0].parametrizations.bias.original.grad)
print("grad of bias score")
print(model[0].parametrizations.bias[0].scores.grad)

opt.step()
print("after opt parameters")
for k, x in model.named_parameters():
    print(k, x)
