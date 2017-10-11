import sys
import numpy as np
from copy import deepcopy
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
        noise: add gaussian noise to gradient to 
            [make your SGD more stochastic](http://deliprao.com/archives/153)
    """
    param[G] += np.random.normal(0, noise / (1 + T)**0.55, param[G].shape)
    
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
    """h = tanh(x)"""
    def __init__(self) :
        return
    def fprop(self, input, output) :
        output[:] = np.tanh(input[0])
    def bprop(self, input, output, out_derivative, gradient):
        gradient[0] += out_derivative * (1 - output * output)

    def get_parameters(self, params) :
        return

class Sigmoid(Layer) :
    """h = 1 / (1 + exp(-x))"""
    def __init__(self) :
        return
    def fprop(self, input, output) :
        output[:] = 1.0 / (1 + np.exp(- input[0]))
    def bprop(self, input, output, out_derivative, gradient):
        gradient[0] += out_derivative * output * (1.0 - output)
    def get_parameters(self, params) :
        return

class Softmax(Layer) :
    """h = 1 / (1 + exp(-x))"""
    def fprop(self, input, output):
        output[:] = np.exp((input[0] - input[0].max())) # subtract constant before exp to avoid float overflow
        output /= output.sum()
        
    def bprop(self, input, output, out_derivative, gradient):
        gradient[0][:] = output
        gradient[0][input[1]] -= 1
    def get_parameters(self, params) :
        return

class LookupLayer() :
    """Stores symbol embeddings"""
    def __init__(self, dimension) :
        self.table = {}         # {word : embedding}
        self.dim = dimension    # dimension of embeddings
        self.active = set()     # set of activated embeddings -> only update those at each SGD step
    
    def __getitem__(self, key) :   # get embedding, v_unknown if word is unknown
        if key not in self.table :
            key = UNKNOWN
        self.active.add(key)
        return self.table[key]
    
    def __call__(self, key) :      # get embedding, create new embedding if word is unknown
        if key not in self.table :
            self.table[key] = [(np.random.random(self.dim) - 0.5) * 2/100, np.zeros(self.dim), np.zeros(self.dim), "E {}".format(key)]
        self.active.add(key)
        return self.table[key]

    def update(self, lr, T, noise) :    # update active parameters and reset set of active parameters
        params = [ self.table[k] for k in self.active ]
        for param in params :
            update(param, lr, T, noise)
        self.active = set()
    
    def get_parameters(self, params) :
        for key in self.active : 
            params.append(self.table[key])
    
    def average(self, T):
        for word in self.table :
            average(self.table[word], T)


class NeuralNet :
    def __init__(self, classes, n_features, hidden_size=16, hidden_layers=2, edim_size=16, lr=0.01, dc=1e-6, noise=0.01):
        self.hidden_size = hidden_size      # taille des couches cachées
        self.hidden_layers = hidden_layers  # nombre de couche cachée
        self.edim_size = edim_size          # dimension des embeddings
        self.out_size = len(classes)        # nombre de classes
        self.n_features = n_features        # taille de la fenêtre
        self.classes = classes              # liste des classes
        self.y_code = {c : i for i,c in enumerate(self.classes)} # codage des classes sur des entiers
        self.lr = lr                        # learning rate
        self.dc = dc                        # decrease constant for learning rate
        self.noise = noise                  # constant for gaussian noise
        
        self.lu = LookupLayer(edim_size)
        self.lu(UNKNOWN)                    # adds UNKNOWN pseudo-word to lookup table
        self.lu.update(0, 1, 0)

        self.layers = [MultipleLinearLayer(n_features, [edim_size for i in range(n_features)], hidden_size),
                       ReLU(), 
                       AffineLayer(hidden_size, hidden_size),
                       ReLU(),
                       AffineLayer(hidden_size, self.out_size),
                       Softmax()]

        self.states = [np.zeros(hidden_size), np.zeros(hidden_size), np.zeros(hidden_size), np.zeros(hidden_size), np.zeros(self.out_size), np.zeros(self.out_size)]  ## NN states
        self.dstates = [np.zeros(hidden_size), np.zeros(hidden_size), np.zeros(hidden_size), np.zeros(hidden_size), np.zeros(self.out_size), np.zeros(self.out_size)] ## their gradient
        
        self.params =[]
        for l in self.layers :
            l.get_parameters(self.params)

    def average(self, T):
        """Returns a fresh copy of this network with parameters averaged"""
        nn = NeuralNet(self.classes, self.n_features, 
                       hidden_size=self.hidden_size,
                       hidden_layers=self.hidden_layers,
                       edim_size=self.edim_size,
                       lr=self.lr,
                       dc=self.dc,
                       noise=self.noise)
        for i in range(len(nn.params)):
            for j in range(len(nn.params[i]) - 1) :
                nn.params[i][j][:] = self.params[i][j][:]
        
        nn.params = []
        for l in nn.layers :
            l.get_parameters(nn.params)
        
        for p in nn.params:
            average(p, T)
        
        
        nn.lu.table = {word : [deepcopy(v) for v in self.lu.table[word]] for word in self.lu.table}
        nn.lu.average(T)
        
        return nn

    
    def forward(self, input, target, pred = False) :
        """Forward pass
            input: list of features
            target: gold class 
            pred: True if test time, False if train time
                At training time: create new embeddings on the fly
                for unseen word
                At test time: use the embedding for UNKNOWN
        """
        target = self.y_code[target]
        if pred :
            in_vecs = [self.lu[key][V] for key in input]
        else :
            in_vecs = [self.lu(key)[V] for key in input]
        
        self.layers[0].fprop(in_vecs, self.states[0])
        for i in range(1, len(self.layers)) :
            self.layers[i].fprop([self.states[i-1]], self.states[i])
        
        return - np.log(self.states[-1][target])

    def backward(self, input, target) :
        """Backward pass
            input: list of features
            target: gold class 
        """
        target = self.y_code[target]
        embeddings = [ self.lu(key) for key in input ]
        in_vecs  = [ e[V] for e in embeddings ]
        din_vecs = [ e[G] for e in embeddings ]
        
        for s in self.dstates :
            s.fill(0.0)
        
        self.layers[-1].bprop([self.states[-2], target], self.states[-1], self.dstates[-1], [self.dstates[-2]])
        for i in reversed(range(1, len(self.layers)-1)) :
            self.layers[i].bprop([self.states[i-1]], self.states[i], self.dstates[i], [self.dstates[i-1]])

        self.layers[0].bprop(in_vecs, self.states[0], self.dstates[0], din_vecs)
    
    def get_learning_rate(self, T):
        """"Learning rate scheduling: learning rate decrease
        as a function of the number T of updates"""
        return self.lr * (1 + T * self.dc)**(-1)
    
    def update(self, T) :
        """Update parameters with the gradients accumulated so far"""
        step_size = self.get_learning_rate(T)
        
        for p in self.params :
            update(p, step_size, T, self.noise)
        self.lu.update(step_size, T, self.noise)
    
    def predict(self) :
        return self.classes[np.argmax(self.states[-1])]

def read_conllu_tagging_data(filename, max_sent=None):
    """
    Lit un corpus au format conll et renvoie une liste de couples.
    Chaque couple contient deux listes de même longueur :
        la première contient les mots d'une phrase
        la seconde contient les tags correspondant
    Par exemple :
    
        [(["le", "chat", "dort"],["DET", "NOUN", "VERB"]),
          (["il", "mange", "."],["PRON", "VERB", "."]),
         ...etc
         ]

    """
    count = 0
    instream = open(filename, "r", encoding="utf8")
    loc = instream.read()
    instream.close()
    sentences = []
    for sent_str in loc.strip().split('\n\n'):

        if max_sent is not None and max_sent < count:
            break

        count += 1
        lines = [line.split() for line in sent_str.split('\n') if line[0] != "#"]
        words = []
        tags = []
        for i, word, _, coarse_pos, _, _, _, _, _, _ in lines:
            if "-" not in i :
                words.append(word)
                tags.append(coarse_pos)
        sentences.append((words,tags))
    return sentences


def get_features(sentence, i) :
    """Returns window of words"""
    res = []
    for shift in range(-2, 3) : # TODO: use parameter to set size of window
        if i + shift < 0 or i + shift >= len(sentence):
            res.append("<s{}>".format(shift))
        else :
            res.append(sentence[i+shift])
    assert(len(res) == 5)
    return res

def evaluate(net, data) :
    acc = 0.0
    tot = 0.0
    loss = 0.0
    for words, tags in data :
        for i in range(len(words)) :
            f = get_features(words, i)
            loss += net.forward(f, tags[i], True)
            y_hat = net.predict()
            if y_hat == tags[i] :
                acc += 1
            tot += 1
    return acc * 100 / tot, loss / tot

def train(net, train, dev, epoch) :
    T = 1
    for e in range(epoch) :
        loss = 0.0
        n = 0
        for words, tags in train :
            for i in range(len(words)) :
                f = get_features(words, i)
                net.forward(f, tags[i])
                net.backward(f, tags[i])
                net.update(T)
                
                T += 1
            n += 1
            if n % 10 == 0 :
                print("\rEpoch {} phrase {} ({}%)".format(e, n, 100.0 * n / len(train)), end="")
                sys.stdout.flush()
        
        avg_net = net.average(T)
        
        acc_train, loss_train = evaluate(avg_net, train[:len(dev)])
        acc_dev,   loss_dev   = evaluate(avg_net, dev)
        print("\rEpoch {} avg: acc train = {} loss train = {}, acc dev = {} loss dev = {} learning rate = {}".format(e, round(acc_train, 4), round(loss_train, 4), round(acc_dev, 4), round(loss_dev, 4), round(avg_net.get_learning_rate(T), 4)), end = " ")
            
        acc_train, loss_train = evaluate(net, train[:len(dev)])
        acc_dev,   loss_dev   = evaluate(net, dev)
        print("(Without averaging: acc train = {} loss train = {}, acc dev = {} loss dev = {})".format(round(acc_train, 4), round(loss_train, 4), round(acc_dev, 4), round(loss_dev, 4)))
        
    

def extract_classes(data) :
    res = set()
    for _, tags in data :
        for t in tags :
            res.add(t)
    return list(res)

def gradient_check(nn, f, target, epsilon = 1e-6) :
    """
        Performs a gradient check on input f (list of features)
        Target: gold class
        epsilon: empirical derivative is computed by the formula
            g(p + epsilon) - g(p - epsilon)
            -------------------------------
                    2 * epsilon
            
            for each parameter p
    """
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

def main() :
    import argparse

    usage = """Simple window-based POS tagger implemented with numpy"""
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("train", type = str, help="Training set (Conll-U format")
    parser.add_argument("dev", type = str, help="Development set (Conll-U format")
    
    parser.add_argument("--learningrate","-l", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--decreaseconstant","-d", type=float, default=1e-6, help="Decrease constant for learning rate scheduling")
    parser.add_argument("--gaussiannoise","-g", type=float, default=0.01, help="Constant for adding gaussian noise to gradient")
    parser.add_argument("--hiddenlayersize","-H", type=int, default=16, help="Number of units in hidden layers")
    parser.add_argument("--embeddingsize","-e", type=int, default=16, help="Size of word embeddings")
    parser.add_argument("--epochs","-i", type=int, default=16, help="Number of epochs")
    

    args = parser.parse_args()

    trainfile = args.train
    devfile = args.dev
    
    traindata = read_conllu_tagging_data(trainfile)
    devdata = read_conllu_tagging_data(devfile)
    
    classes = extract_classes(traindata)
    
    nn = NeuralNet( classes,
                    5,
                    hidden_size=args.hiddenlayersize,
                    hidden_layers=2,
                    edim_size=args.embeddingsize,
                    lr=args.learningrate,
                    dc=args.decreaseconstant,
                    noise=args.gaussiannoise)
    
    train(nn, traindata, devdata, args.epochs)


main()





