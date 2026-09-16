class Linear:
    def __init__(self, fan_in, fan_out, bias = True):
        self.weight = torch.randn((fan_in, fan_out), generator = g)
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight 
        if self.bias is not None:
            self.out += self.bias 
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum = 0.1):
        self.dim = dim 
        self.eps = eps
        self.momentum = momentum
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        if self.training: 
            xmean = x.mean(0, keepdim = True)
            xvar = x.var(0, keepdim = True)
        else: 
            xmean = self.running_mean
            xvar = self.running_var

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum)*self.running_mean + momentum*xmean
                self.running_var = (1 - self.momentum)*self.running_var + momentum*xvar
        
        xhat = (x - xmean)/torch.sqrt(xvar + self.eps)
        self.out = xhat*self.gamma + self.beta 
        return self.out 

    def parameters(self):
        return [self.gamma, self.beta]
    
class Tanh:
    
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []

n_embd = 10
n_hidden = 1000
vocab_size = 30

C = torch.randn((vocab_size, n_embd), generator = g)
layers = [
    Linear(n_embd*block_size, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
]

     
with torch.no_grad():
    layers[-1].weights *= 0.1 #???
    for layer in layers[:-1]:
        if layer isinstance(layer, Linear):
            layer.weights *= 5/3
    
parameters = [C] + [p for layer in layers for p in layer.parameters()]
sum_p = sum(p.nelement() for p in parameters)

for p in parameters:
    p.requires_grad = True

batch_size = 32
max_steps = 200000
lossi =[]

for _ in range(max_steps):
    #create batches
    ix = torch.randn(0, Xtr.shape[0], (batch_size, ), generator = g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    #forward pass
    embd = C[Xb]
    embd_cat = embd.view(embd.shape[0], -1)
    x = embdcat
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #update parameters
    lr = 0.1 for i < 100000 else 0.01
    for p in parameters:
        p.data *= -lr*p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item:.4f}')
    lossi.append(loss.log10().item())
    
    

    
