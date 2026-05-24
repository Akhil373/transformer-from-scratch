# %% Cell 1
import math
import random


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):  # this does other + self (eg. 2 + a)
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):  # this does other * self (eg. 2 * a)
        return self * other

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), "-")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other):  # this does other - self (eg. 2 - a)
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers are supported"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b
e.label = "e"
c = Value(data=10.0, label="c")
d = e + c
d.label = "d"
f = Value(-2.0, label="f")
L = f * d
L.label = "L"

L.grad = 1.0  #  dL/dL = 1
f.grad = 4.0  #  dL/df = 4.0 (value of d)
d.grad = -2.0  # dL/dd = -2.0 (value of  f)

c.grad = -2.0  # dL/dc = dL/dd * dd/dc
e.grad = -2.0  # dL/de = dL/dd * dd/de

a.grad = 6.0  # dL/da = dL/de + de/da
b.grad = -4.0  # dL/db = dL/de  + de/db

# %% Cell 1.5
a = Value(2.0)
print(3 * a)


# %% Cell 2
def print_graph(v, indent=0):
    print(
        "  " * indent
        + f"Label: {v.label}, Data: {v.data:.2f}, Grad: {v.grad:.2f}, Op: {v._op}"
    )

    for child in v._prev:
        print_graph(child, indent + 2)


print_graph(L)
# %% Cell 3
h = 0.01
a.data += h * a.grad
b.data += h * b.grad
c.data += h * c.grad
f.data += h * f.grad

e = a * b
d = c + e
L = d * f
L.data


# %% Cell 4
def scribble():
    h = 0.001
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a * b
    e.label = "e"
    c = Value(data=10.0, label="c")
    d = c + e
    d.label = "d"
    f = Value(-2.0, label="f")
    L = d * f
    L.label = "L"
    L1 = L.data

    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    b.data += h
    c = Value(10.0, label="c")
    e = a * b
    e.label = "e"
    c = Value(data=10.0, label="c")
    d = c + e
    d.label = "d"
    f = Value(-2.0, label="f")
    L = d * f
    L.label = "L"
    L2 = L.data
    print((L2 - L1) / h)


scribble()
# %% Cell 5

# inputs
x1 = Value(data=2.0, label="x1")
x2 = Value(data=0.0, label="x2")

# weights
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

# bias
b = Value(6.8813735870195, label="b")

# x1w1 + x2w2 + b
x1w1 = x1 * w1
x1w1.label = "x1w1"

x2w2 = x2 * w2
x2w2.label = "x2w2"

x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1w1x2w2"

n = x1w1x2w2 + b
n.label = "n"
o = n.tanh()
o.label = "o"

# %% Cell 6
o.backward()

# %% Cell 7
o.grad = 1.0
o._backward()
n._backward()
b._backward()
x1w1x2w2._backward()
x1w1._backward()
x2w2._backward()

# %% Cell 8
print_graph(o)

# %% Cell 9
import torch

# %% Cell 10
x1 = torch.Tensor([2.0]).double()
x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()
x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double()
w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()
w2.requires_grad = True
b = torch.Tensor([6.8813735870195]).double()
b.requires_grad = True
n = x1 * w1 + x2 * w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()
print("-" * 10)
print("x2", x2.grad.item())
print("w2", w2.grad.item())
print("x1", x1.grad.item())
print("w1", w1.grad.item())


# %% Cell 11
class Neuron:
    def __init__(self, nin) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# %% Cell 10.2
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])  # 3 inputs, two hidden layers of 4 neurons, 1 output
print(n(x))
len(n.parameters())

# %% Cell 11
# example dataset
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets


# %% Cell 12
for k in range(20):
    # forward pass
    ypreds = [n(x) for x in xs]
    loss = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, ypreds)])

    # zero grad
    for p in n.parameters():
        p.grad = 0.0

    # backward pass
    loss.backward()

    # gradient descent
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(f"Step: {k + 1}, loss: {loss.data}")

# %% Cell
ypreds
