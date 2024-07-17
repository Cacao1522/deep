
import torch
import torch.nn as nn
from torch.nn import functional as F

x = torch.tensor([1., 2., 3.])

y = torch.tensor([3., 2., 1.])

M = torch.tensor([[1., 2., 3.], [3., 2., 1.]])

N = torch.ones(2, 4, 3)

K = torch.arange(12).view(4,3) # viewは配列サイズを4x3に

A = torch.ones(2, 4, 5)

V = torch.ones(2, 5, 3)
V[1,:,:] = 2.

# 要素への代入はnumpyと同様
# x[0] = 2.
N[0,0,1] = 12.

# スライスもnumpyと同様
print("M[:,0] =", M[:,0])

# (a)
print("--------------------------------------------------")
print("(a)")

print("x + y =", x + y)

print("x - y =", x - y)

print("x * y =", x * y)

print("x / y =", x / y)

# (b)
print("--------------------------------------------------")
print("(b)")

print("M + x =", M + x)

print("M - x =", M - x)

print("M * x =", M * x)

print("M / x =", M / x)

# (c)
print("--------------------------------------------------")
print("(c)")

print("N + K =", N + K)

# (d)
print("--------------------------------------------------")
print("(c)")

print("N @ M.transpose(0,1) =", N @ M.transpose(0,1))

linear = nn.Linear(in_features=3, out_features=2, bias=False)

print("linear.weight:", linear.weight)

print("linear(M) =", linear(M))

print("linear(N) =", linear(N))

# (e)
print("--------------------------------------------------")
print("(e)")

print("A @ V =", A @ V)

# (f)
print("--------------------------------------------------")
print("(f)")

print("V.transpose(1,2) =", V.transpose(1,2))
print("N @ V.transpose(1,2) =", N @ V.transpose(1,2))

# (g)
print("--------------------------------------------------")
print("(g)")

print("F.softmax(N, dim=0) =", F.softmax(N, dim=0))
print("F.softmax(N, dim=1) =", F.softmax(N, dim=1))
print("F.softmax(N, dim=2) =", F.softmax(N, dim=2))
