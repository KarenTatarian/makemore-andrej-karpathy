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

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

#MLP revisited
g = torch.Generator().manual_seed(2147483647)
n_embd = 10 #the dimensionality of the character embedding vectors
n_hidden = 200 #number of neurons in the hidden layer of MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/(n_embd*block_size)**0.5 #0.2
#b1 = torch.randn(n_hidden,                        generator=g) 
#* 0.01 #setting it very small to get a little bit of entropy (variation and diversity in initialization to help optimization)
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01
b2 = torch.randn(vocab_size,                      generator=g) * 0 
#equivalent to torch.zeros(vocab_size), zero initialization does not mean it is frozen, b2 still gets trained
 
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

parameters = [C, W1, W2, b2, bngain, bnbias]

print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

max_steps = 2000 #200000
batch_size = 32
lossi = []

for i in range(max_steps):
    #minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator = g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    #forward pass
    emb = C[Xb] #embed the characters into vector
    embcat = emb.view(emb.shape[0], -1) #concatenate the vectors

    #Linear layer
    hpreact = embcat @ W1 #+ b1 #hidden layer pre-activation layer

    #batch normalization 
    bnmeani = hpreact.mean(0, keepdim = True)
    bnstdi = hpreact.std(0, keepdim = True) + 0.0000001 #added epsilon, not needed for this use case
    hpreact = bngain*(hpreact - bnmeani)/bnstdi + bnbias

    with torch.no_grad():
        bnmean_running = 0.999*bnmean_running + 0.001*bnmeani #0.001 here is the momentum
        bnstd_running = 0.999*bnstd_running + 0.001*bnstdi

    h = torch.tanh(hpreact) #non-linear activation layer 

    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 1000 else 0.01
    #update parameters
    for p in parameters:
        p.data += -lr*p.grad 
    
    #track stats
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item(): .4f}')
    lossi.append(loss.log10().item())

plt.plot(lossi)
plt.show()

"""
#calibrate the batch norm at the end of training 
#instead of doing that, it can be calculated while running the training 
#no longer need this because bnmean is almost equal to bnmean_running 
#same for bnstd_running 
with torch.no_grad():
    #pass the training set through
    emb = C[Xtr]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    #measure the mean/std over the entire the training set 
    bnmean = hpreact.mean(0, keepdim = True)
    bnstd = hpreact.std(0, keepdim = True)

"""

@torch.no_grad() # this decorator disables gradient tracking and it tells pytorch to not require grad and we will not be performing backward
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  emb = C[x] # (N, block_size, n_embd)
  embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
  hpreact = embcat @ W1 #+ b1
  #hpreact = bngain*(hpreact - hpreact.mean(0, keepdim = True))/hpreact.std(0, keepdim = True) + bnbias
  #hpreact = bngain*(hpreact - bnmean)/bnstd + bnbias
  hpreact = bngain*(hpreact - bnmean_running)/bnstd_running + bnbias
  h = torch.tanh(hpreact) # (N, n_hidden)
  logits = h @ W2 + b2 # (N, vocab_size)
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')

#testing some word generations 
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0]*block_size 
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1) #+ b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim = 1)
        ix = torch.multinomial(probs, num_samples = 1, generator = g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))

#---BRINGING IT TOGETHER-----

class Linear:

    def __init__(self, fan_in, fan_out, bias = True):
        self.weight = torch.randn((fan_in, fan_out), generator = g)/fan_in**0.5 
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

n_embd = 10 #the dimensionality of the character embedding vectors 
n_hidden = 100 # number of neurons in the hidden layer of the MLP
g = torch.Generator().manual_seed(2147483647) #for reproducibiltiy 

C = torch.randn((vocab_size, n_embd), generator = g)
layers = [
    Linear(n_embd*block_size, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size), BatchNorm1d(vocab_size),
]

with torch.no_grad():
    #last layer: make less confident (softmax layer)
    #layers[-1].weight *= 0.1
    layers[-1].gamma *= 0.1 #since now the last layer is batchNorm1d, it would be .weight if the last layer was linear
    #for other layers: apply gain 
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3
            #Tanh is trying to squash the values to 0 so after several layers the std get smaller and smaller and closer to 0
            #to off-set the squashing and keep a std close to 1 throughout the layers 
            #we give the linear layers a gain to offset the squashing (unsure where specifically 5/3 originated from but it works well)
            #more than the saturation becomes way too high meaning that the tanh outputs are too close to -1 and +1 
            #this is bad because if outputs are too close to +/- 1 then the derivative is very close to almost 0
            #then backpropagation has very little signal to tell the earlier weights how they should change
            #we generally want the Tanh outputs to be spread more around the middle (near 0) to give a great gradient

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) #total params
for p in parameters:
    p.requires_grad = True

#Optimization 
max_steps = 200000
batch_size = 32
lossi = []
ud = []

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size, ), generator = g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    #forward pass 
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1) #concatenate the vectors
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    #backward pass
    for layer in layers:
        layer.out.retain_grad() #AFTER_DEBUG: would take out retain_graph
    for p in parameters:
        p.grad = None
    loss.backward()

    #update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr*p.grad

     #track stats
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item(): .4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([(lr*p.grad.std()/p.data.std()).log10().item() for p in parameters])
    
    if i >= 1000:
        break #just for debugging purposes


# visualize histograms
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
  if isinstance(layer, Tanh):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends);
plt.title('activation distribution')
plt.show()

# visualize histograms
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
  if isinstance(layer, Tanh):
    t = layer.out.grad
    print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends);
plt.title('gradient distribution')
plt.show()


# visualize histograms
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i,p in enumerate(parameters):
  t = p.grad
  if p.ndim == 2:
    print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('weights gradient distribution');
plt.show()


plt.figure(figsize=(20, 4))
legends = []
for i,p in enumerate(parameters):
  if p.ndim == 2:
    plt.plot([ud[j][i] for j in range(len(ud))])
    legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
#if this is much higher than that, then the update is happening too fast
#if it is below that, then the parameters are learning too slow (==> solution: increase learning rate)
plt.legend(legends);
plt.show()
