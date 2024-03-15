import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


    def parameters(self):
        return []


    def cross_entropy_loss(self, scores, y_train, log_epsilon = 0.000001):
        total_loss = 0
        for s_row, y_row in zip(scores, y_train):
            total_loss += -sum(yi * si.log(log_epsilon) for si, yi in zip(s_row, y_row))
        return total_loss

    def optimize(self, loss, learning_rate = 0.01):
        self.zero_grad()
        loss.backward()
        for p in self.parameters():
            p.data -= learning_rate * p.grad

    def accuracy(self, scores, y_train):
        acc = []
        for s_row, y_row in zip(scores, y_train):
            acc.append(sum((yi * 100 * (1 - (yi - si))) for si, yi in zip(s_row, y_row)))
        return sum(acc) / len(acc)

class Neuron(Module):

    def __init__(self, nin, index, relu=False, softmax=False):
        if softmax == True:
            self.w = [Value(0.0)]
        else:
            self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.index = index
        self.relu = relu
        self.softmax = softmax

    def __call__(self, x):
        if self.softmax:
            return self._softmax(x)
        else:
            act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
            return act.relu() if self.relu else act

    def _softmax(self, x):
        max_x = x[0]
        for xi in x:
            if xi >= max_x:
                max_x = xi
        exp_x = [(xi - max_x).exp() for xi in x]

        return exp_x[self.index] / sum(exp_x)


    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'Softmax' if self.softmax else 'ReLU' if self.relu else 'Linear'}Neuron[{self.index}]({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, index, **kwargs) for index in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out


    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    # It will automatically add softmax layer, so for example
    # if one needs a network with 10 input neurons, 6 neurons in first hidden layer
    # 6 neurons in second hidden layer and 4 neurons in the output layer it will be:
    # MLP(10, [6, ,6, 4[). The last layer will have softmax values.
    def __init__(self, nin, nouts, softmax=True, relu=False):
        if softmax:
            nouts = nouts + [nouts[-1]]
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1],
                             relu = relu,
                             softmax = (softmax == True and (i == len(nouts) - 1))) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
