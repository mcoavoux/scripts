import sys
import numpy as np
np.random.seed(0)

V,G,H,NAME=list(range(4))       # V: parameter, G: its gradient, H: cahed parameters for averaging, NAME: string 
UNKNOWN="UNKNOWN"

def xavier(outsize, insize) :
    """Initialize a weight matrix with Xavier Glorot's method (see Goldberg 2015's primer)"""
    tmp = np.random.random((outsize, insize)) - 0.5
    tmp *= 2
    return  tmp * (6 / (outsize + insize))**0.5

def update(param, lr, T, noise) :
    """
    Update parameter,
        lr: learning rate
        T: for averaging
        noise: TODO
    """
    param[V] -= lr * param[G]
    param[H] -= lr * T * param[G]
    param[G].fill(0)

def average(param, T) :
    param[V] -= (1.0/T) * param[H]

def increment(param, i, epsilon) : # for gradient checking, add epsilon to one coefficient
    if len(param[V].shape) == 1 :
        param[V][i] += epsilon
    else : 
        l,c = param[V].shape
        param[V][i//c,i%c] += epsilon

def set_empirical_gradient(param, i, value) : # for gradient checking, param[H] is used to store empirical gradient
    if len(param[V].shape) == 1 :
        param[H][i] = value
    else : 
        l,c = param[V].shape
        param[H][i//c,i%c] = value
    

class Layer :
    """Abstract class for an NN layer"""
    def fprop(self, input, output): # forward
        assert False
    def bprop(self, input, output, out_derivative, gradient): # backward
        assert False
    def get_parameters(self, params) : # parameters for this layer
        assert False

class LinearLayer(Layer) :
    """h = W x"""
    def __init__(self, insize, outsize) :
        self.w = [xavier(outsize, insize), np.zeros((outsize, insize)), np.zeros((outsize, insize)), "Linear w"]
        
    def fprop(self, input, output) :
        output[:] = self.w[V].dot(input[0])
    
    def bprop(self, input, output, out_derivative, gradient):
        self.w[G] += np.outer(out_derivative, input[0])
        gradient[0] += self.w[V].transpose().dot(out_derivative)
    
    def get_parameters(self, params) :
        params.append(self.w)

class AffineLayer(Layer) :
    """h = W x + b"""
    def __init__(self, insize, outsize) :
        self.b = [np.zeros(outsize), np.zeros(outsize), np.zeros(outsize), "Affine b"]
        self.w = [xavier(outsize, insize), np.zeros((outsize, insize)), np.zeros((outsize, insize)),  "Affine w"]
    
    def fprop(self, input, output) :
        output[:] = self.w[V].dot(input[0]) + self.b[V]
        
    def bprop(self, input, output, out_derivative, gradient):
        self.b[G] += out_derivative
        self.w[G] += np.outer(out_derivative, input[0])
        gradient[0] += self.w[V].transpose().dot(out_derivative)

    def get_parameters(self, params) :
        params.append(self.b)
        params.append(self.w)

class MultipleLinearLayer(Layer) :
    """h = W1 x1 + W2 x2 + ... Wn xn + b"""  # used when input is a list of vectors (e.g. embeddings)
    def __init__(self, insize, insizes, outsize) :
        self.layers = [LinearLayer(insizes[i], outsize) for i in range(insize)]
        self.b = [np.zeros(outsize), np.zeros(outsize), np.zeros(outsize),  "Multiple b"]
        self.buffer = np.zeros(outsize)
    
    def fprop(self, input, output) :
        output[:] = self.b[V]
        for i,l in enumerate(self.layers) :
            l.fprop([input[i]], self.buffer)
            output[:] += self.buffer
        
    def bprop(self, input, output, out_derivative, gradient):
        self.b[G] += out_derivative
        for i,l in enumerate(self.layers) :
            l.bprop([input[i]], output, out_derivative, [gradient[i]])

    def get_parameters(self, params) :
        params.append(self.b)
        for l in self.layers :
            l.get_parameters(params)
    

class ReLU(Layer) :
    """h = max(x, 0)"""
    def __init__(self) :
        return
    def fprop(self, input, output) :
        output[:] = input[0] * (input[0] > 0)
    def bprop(self, input, output, out_derivative, gradient):
        gradient[0][:] += out_derivative * (input[0] > 0)

    def get_parameters(self, params) :
        return

class Tanh(Layer) :
    def __init__(self) :
        return
    def fprop(self, input, output) :
        output[:] = np.tanh(input[0])
    def bprop(self, input, output, out_derivative, gradient):
        gradient[0] += out_derivative * (1 - output * output)

    def get_parameters(self, params) :
        return

class Sigmoid(Layer) :
    def __init__(self) :
        return
    def fprop(self, input, output) :
        output[:] = 1.0 / (1 + np.exp(- input[0]))
    def bprop(self, input, output, out_derivative, gradient):
        gradient[0] += out_derivative * output * (1.0 - output)
    def get_parameters(self, params) :
        return

class Softmax(Layer) :
    def fprop(self, input, output):
        output[:] = np.exp((input[0] - input[0].max()))
        output /= output.sum()
        
    def bprop(self, input, output, out_derivative, gradient):
        gradient[0][:] = output
        gradient[0][input[1]] -= 1
    def get_parameters(self, params) :
        return

class XorNet :
    def __init__(self, hidden_size, num_hidden, lr) :
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden
        self.lr = lr
    
        self.layers = [AffineLayer(2, self.hidden_size), Sigmoid()]
        for i in range(self.num_hidden-1) :
            self.layers.append(AffineLayer(self.hidden_size, self.hidden_size))
            self.layers.append(Sigmoid())
        self.layers.append(AffineLayer(self.hidden_size, 2))
        self.layers.append(Softmax())
        
        self.states = [np.zeros(2)] + [np.zeros(self.hidden_size) for _ in range(num_hidden*2)]  + [np.zeros(2), np.zeros(2)]
        self.dstates = [np.zeros(2)] + [np.zeros(self.hidden_size) for _ in range(num_hidden*2)]  + [np.zeros(2), np.zeros(2)]
        
    
        self.params = []
        for l in self.layers :
            l.get_parameters(self.params)
    
    def forward(self, input_vec, target) :
        self.states[0] = input_vec
        
        for i,l in enumerate(self.layers) :
            l.fprop([self.states[i]], self.states[i+1])
        
        #   si target == 1  -> log P(X=1), si target == 0 --> log P(X = 0) = log (1 - P(X = 1))
        return - np.log(self.states[-1][target])
    
    def backward(self, input, target):
        
        for d in self.dstates :
            d.fill(0.0)
        
        self.layers[-1].bprop([[self.states[-2]], target], self.states[-1], self.dstates[-1], [self.dstates[-2]])
        
        for i in reversed(range(1, len(self.layers)-1)) :
            self.layers[i].bprop([self.states[i]], self.states[i+1], self.dstates[i+1], [self.dstates[i]])
    
    def update(self, lr, T, noise) :
        for p in self.params :
            update(p, lr, T, noise)
    
    def predict(self) :
        return np.argmax(self.states[-1])
    
    def train_one_epoch(self, data, targets) :
        loss = 0
        for x,y in zip(data, targets) :
            loss += self.forward(x, y)
            self.backward(x, y)
            self.update(self.lr, 0, 0)
        return loss / len(data)
    
    def evaluate(self, data, targets) :
        loss = 0
        acc = 0
        for x,y in zip(data, targets) :
            loss += self.forward(x, y)
            y_hat = self.predict()
            if y_hat == y :
                acc += 1
        
        
        return loss / len(data), acc / len(data)
        


def gradient_check(nn, f, target, epsilon = 1e-6) :
    loss = nn.forward(f, target)
    nn.backward(f, target)
    
    nn.lu.get_parameters(nn.params)
    
    for p in nn.params :
        for i in range(p[V].size) :
            increment(p, i, epsilon)
            a = nn.forward(f, target)
            increment(p, i, -epsilon)
            increment(p, i, -epsilon)
            c = nn.forward(f, target)
            increment(p, i, epsilon)
            set_empirical_gradient(p, i, (a - c) / (2*epsilon))
        v = np.abs(p[G] - p[H]).sum() / p[V].size
        print(p[NAME])
        print("p {}".format(v))


def boolean_test():
    
    import itertools
    
    data = [np.array(v) for v in [[-1,-1], [-1, 1], [1,-1], [1,1]]]
    
    for targets in itertools.product([0,1], repeat = 4) :
        net = XorNet(100, 2, 0.1)
        
        loss, acc = 0, 0
        epoch = 0
        
        print("Learning {} ...".format(targets, epoch))
        while acc < 1.0 :
            net.train_one_epoch(data, targets)
            newloss, newacc = net.evaluate(data, targets)
            
            epoch += 1
            loss, acc = newloss, newacc
            if epoch % 10000 == 0 :
                print("Epoch {} : loss = {} acc = {}".format(epoch, loss, acc))
            
        print("Learned {} in {} epochs".format(targets, epoch))



boolean_test()





