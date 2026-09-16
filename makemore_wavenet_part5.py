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
block_size = 3 #context window size

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
        if self.training:
            xmean = x.mean(0, keepdim = True) #batch mean
            xvar = x.var(0, keepdim = True, unbiased = True) #batch variance
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
torch.manual_seed(42); # seed rng for reproducibility

n_embd = 10 #the dimensionality of the character embedding vectors
n_hidden = 200 #number of neurons in the hidden layer of the MLP

C = torch.rand((vocab_size, n_embd))
layers = [
    Linear(n_embd * block_size, n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
]

with torch.no_grad():
    #last layer: make less confident (softmax layer)
    layers[-1].weight *= 0.1 

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) #total params
for p in parameters:
    p.requires_grad = True


#Optimization 
max_steps = 200000
batch_size = 32
lossi = []

#optimizing with stochastic gradient descent
for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size, ))
    Xb, Yb = Xtr[ix], Ytr[ix]

    #forward pass 
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1) #concatenate the vectors
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

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

plt.plot(lossi)
plt.show()

plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1)) #pytorch rearanges the array of floats into 2D tensor of 1000 columns and it can figure out the and then average across the rows --> to get a shape of [200]
#put the layers into eval mode (needed for batchnorm especially)
for layer in layers:
    layer.training = False

# evaluate the loss
@torch.no_grad() # this decorator disables gradient tracking inside pytorch
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  emb = C[x] #(N, block_size, n_embd)
  x = emb.view(emb.shape[0], -1) #conat into (N, block_size*n_embd)
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')


# sample from the model
for _ in range(20):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
        # forward pass the neural net
        emb = C[torch.tensor([context])] #(N, block_size, n_embd)
        x = emb.view(emb.shape[0], -1) #conat into (N, block_size*n_embd)
        for layer in layers:
        x = layer(x)
        logits = x
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
