import torch 
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt 

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
#block_size = 3 #context window size
block_size = 8 #context window size

#build dataset
def build_dataset(words): 
    X, Y = [], []

    for w in words:
        context = [0]*block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] #crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1]) #80%
Xdev, Ydev = build_dataset(words[n1:n2]) #10%
Xte, Yte = build_dataset(words[n2:]) #10%


class Linear:

    def __init__(self, fan_in, fan_out, bias = True):
        self.weight = torch.randn((fan_in, fan_out))/fan_in**0.5 
        #fan_in and fan_out here giving the dimensions fan_in x fan_out with mean around 0 and std 1 but then with 1/sqrt(fan_in) distribution is scaled down 
        #the (1/fan_in**0.5) factor can be removed once normalization layers are added everywhere 

        self.bias = torch.zeros(fan_out) if bias else None 

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out 
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:

    def __init__(self, dim, eps = 1e-5, momentum = 0.1): #dim being number of features in 
        self.eps = eps 
        self.momentum = momentum
        self.training = True
        #parameters (trained in backdrop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        #buffers (trained with a running momentum update)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        #calculate the forward pass
        #note x.shape = (batch_size, num_features)
        #    Rows = different training examples
        #    Columns = different neurons/features
        # this is why we calculate mean of each column along the row dimensionality
        #so we end up with xmean having 1D row of all mean values of each feature
        # previously we had designed it for 2D but now since have 3D (2 inputs, batch_size)
        if self.training:
            if x.ndim == 2:
                dim = 0
            if x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim = True) #batch mean #tuples can be passed on so mean is computed over multiple dimensions
            xvar = x.var(dim, keepdim = True, unbiased = True) #batch variance
        else:
            xmean = self.running_mean 
            xvar = self.running_var
        xhat = (x - xmean)/torch.sqrt(xvar + self.eps) #normalize to unit variance
        self.out = xhat*self.gamma + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum)*self.running_mean + self.momentum*xmean
                self.running_var = (1 - self.momentum)*self.running_var + self.momentum*xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.rand(num_embeddings, embedding_dim) # this is what was previously C
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]

class FlattenConsecutive:
    def __init__(self, n):
        self.n = n #number of consecutive 

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n) # when we have // that is an element wise division 
        if x.shape[1] == 1: 
            x = x.squeeze(1) #squeeze in pytorch squeezes any dimension that is one, here we are specifying exactly which dimension to squeeze along 
        #self.out = x.view(x.shape[0], -1)
        self.out = x
        return self.out 
    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out 
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

torch.manual_seed(42); # seed rng for reproducibility

n_embd = 24 #the dimensionality of the character embedding vectors
n_hidden = 128 #number of neurons in the hidden layer of the MLP

#C = torch.rand((vocab_size, n_embd))
#model = Sequential([
#    Embedding(vocab_size, n_embd),
#    Flatten(),
#    Linear(n_embd * block_size, n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
#    Linear(n_hidden, vocab_size),
#])

# hierarchical network
model = Sequential([
  Embedding(vocab_size, n_embd),
  FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, vocab_size),
])


with torch.no_grad():
    #last layer: make less confident (softmax layer)
    model.layers[-1].weight *= 0.1 

#parameters = [p for layer in layers for p in layer.parameters()]
parameters = model.parameters()
print(sum(p.nelement() for p in parameters)) #total params
for p in parameters:
    p.requires_grad = True

#Note: 
# 1) reminder in python: list(range(10))[::2] ==> gives us all the even integers in the list 
#                        list(range(10))[1::2] ==> gives us all the odd integers in the list 
# 2) reminder that right now we have the flatten acting as:
#    e = torch.randn(4, 8, 10) ==> would lead to shape of (4, 80)
# 3) 
#    In x @ a, the last dimension of x must match the first dimension of a.
#    The dimensions before the last one in x are preserved.
#    For example looking at the shapes for x @ a:
#    x: [4, 4, 20] @ [20, 100] -> [4, 4, 100]
#    x: [8, 20]    @ [20, 100] -> [8, 100]
#     So x @ a replaces x's last dimension (20) with a's last dimension (100)
#    this is why we are building FlattenConsecutive so we can input batches of two 
#.   before we were inputing [n_example, block_size, n_embedding] now it would be like number of examples, batch per example, ...

#Optimization 
max_steps = 200000
batch_size = 32
lossi = []

#optimizing with stochastic gradient descent
for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size, ))
    Xb, Yb = Xtr[ix], Ytr[ix]

    #forward pass 
    #emb = C[Xb]
    #x = emb.view(emb.shape[0], -1) #concatenate the vectors
    #x = Xb
    #for layer in layers:
    #    x = layer(x)
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #update: simple SGD
    lr = 0.1 if i < 150000 else 0.01
    for p in parameters:
        p.data += -lr*p.grad

     #track stats
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item(): .4f}')
    lossi.append(loss.log10().item())
    
    #ßbreak

plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
#pytorch rearanges the array of floats into 2D tensor of 1000 columns and it can figure out the and then average across the rows --> to get a shape of [200]
plt.show()


#put the layers into eval mode (needed for batchnorm especially)
for layer in model.layers:
    layer.training = False

# evaluate the loss
#usually you want to look at training and validation loss together 
@torch.no_grad() # this decorator disables gradient tracking inside pytorch
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  #emb = C[x] #(N, block_size, n_embd)
  #x = emb.view(emb.shape[0], -1) #conat into (N, block_size*n_embd)
  #for layer in layers:
  #  x = layer(x)
  logits = model(x)
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')

"""
performance log
original (3 character context + 200 hidden neurons, 12K params): train 2.058, val 2.105
context: 3 -> 8 (22K params): train 1.918, val 2.027
flat -> hierarchical (22K params): train 1.941, val 2.029
fix bug in batchnorm: train 1.912, val 2.022
scale up the network: n_embd 24, n_hidden 128 (76K params): train 1.769, val 1.993
"""

# sample from the model
for _ in range(20):
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
        # forward pass the neural net
        #emb = C[torch.tensor([context])] #(N, block_size, n_embd)
        #x = emb.view(emb.shape[0], -1) #conat into (N, block_size*n_embd)
        #x = torch.tensor([context])
       # for layer in layers:
        #    x = layer(x)
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        # sample from the distribution
        ix = torch.multinomial(probs, num_samples=1).item()
        # shift the context window and track the samples
        context = context[1:] + [ix]
        out.append(ix)
        # if we sample the special '.' token, break
        if ix == 0:
            break
    
    print(''.join(itos[i] for i in out)) # decode and print the generated word
