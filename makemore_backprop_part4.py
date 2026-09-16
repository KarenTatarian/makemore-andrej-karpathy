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

#utility function we will use later when comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

#MLP revisited
g = torch.Generator().manual_seed(2147483647)
n_embd = 10 #the dimensionality of the character embedding vectors
n_hidden = 64 #number of neurons in the hidden layer of MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
#Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd*block_size)**0.5) #0.2
b1 = torch.randn(n_hidden,                        generator=g) * 0.1 #using b1 for fun here 
#* 0.01 #setting it very small to get a little bit of entropy (variation and diversity in initialization to help optimization)
#Layer2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
#equivalent to torch.zeros(vocab_size), zero initialization does not mean it is frozen, b2 still gets trained

# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

# Note: I am initializating many of these parameters in non-standard ways
# because sometimes initializating with e.g. all zeros could mask an incorrect
# implementation of the backward pass.

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

batch_size = 32
n = batch_size # a shorter variable also, for convenience
# construct a minibatch
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

# forward pass, "chunkated" into smaller steps that are possible to backward one at a time
emb = C[Xb] # embed the characters into vectors
embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
# Linear layer 1
hprebn = embcat @ W1 + b1 # hidden layer pre-activation

# BatchNorm layer
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias

# Non-linearity
h = torch.tanh(hpreact) # hidden layer

# Linear layer 2
logits = h @ W2 + b2 # output layer

# cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes # subtract max for numerical stability
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

# PyTorch backward pass
for p in parameters:
  p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, # afaik there is no cleaner way
          norm_logits, logit_maxes, logits, h, hpreact, bnraw,
         bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
         embcat, emb]:
  t.retain_grad()
loss.backward()
loss

"""
My attempted code 
"""
#attempt to backprop manually 
#loss = -logprobs[range(n), Yb].mean()
"""
dloss = 1.0
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = (-1.0/n) * dloss

#logprobs = probs.log()
dprobs = (1.0/probs)*dlogprobs

#probs = counts * counts_sum_inv
dcounts_sum_inv = (counts*dprobs).sum(1, keepdim = True)
dcounts = counts_sum_inv * dprobs

#counts_sum_inv = counts_sum**-1.0 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
dcounts_sum = (-counts_sum**-2)*dcounts_sum_inv

#counts_sum = counts.sum(1, keepdims=True)
dcounts += torch.ones_like(counts) * dcounts_sum  #dcounts_sum #this is the line that I made a mistake in but changing it did not fix the False at norm_logits
#changing it did not change the outcome

#counts = norm_logits.exp()
dnorm_logits = counts * dcounts

#norm_logits = logits - logit_maxes # subtract max for numerical stability
dlogits = dnorm_logits 
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)

# logit_maxes = logits.max(1, keepdim=True).values
max_values, max_indices = torch.max(logits, dim=1, keepdim=True)
dlogits_2 = torch.zeros_like(logits)
dlogits_2.scatter_(1, max_indices, 1.0)
dlogits_2 *= dlogit_maxes
dlogits += dlogits_2

# Linear layer 2 logits = h @ W2 + b2 # output layer
dW2 = h.T @ dlogits 
db2 = dlogits.sum(0)
dh = dlogits @ W2.T

#Non-linearity h = torch.tanh(hpreact) # hidden layer
dhpreact = (1 - h**2)*dh 

#hpreact = bngain * bnraw + bnbias
dbngain = (bnraw * dhpreact).sum(0)
dbnbias = dhpreact.sum(0)#, keepdim = True)
dbnraw = bngain * dhpreact 

#bnraw = bndiff * bnvar_inv
dbnvar_inv = (bndiff*dbnraw).sum(0, keepdim = True)
dbndiff = bnvar_inv * dbnraw

#bnvar_inv = (bnvar + 1e-5)**-0.5 
dbnvar = (-0.5*(bnvar + 1e-5)**-1.5)*dbnvar_inv 

#bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2)*dbnvar 

#bndiff2 = bndiff**2
dbndiff += (2*bndiff)* dbndiff2

#backpropagation bndiff = hprebn - bnmeani 
dbnmeani = -dbndiff.sum(0)#, keepdim = True) 
dhprebn = dbndiff 

# BatchNorm layer
#bnmeani = 1/n*hprebn.sum(0, keepdim=True) 
dhprebn += (1.0/n) * torch.ones_like(hprebn)* dbnmeani

# first input hprebn = embcat @ W1 + b1 
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0) 
dembcat = dhprebn @ W1.T
demb = dembcat.view(emb.shape)

# emb = C[Xb]
dC = torch.zeros_like(C)
dC[Xb] += demb

"""
"""
Result of my solution:
 makemore-andrej-karpathy % python3 makemore_backprop_part4.py
4137
logprobs        | exact: True  | approximate: True  | maxdiff: 0.0
probs           | exact: True  | approximate: True  | maxdiff: 0.0
counts_sum_inv  | exact: True  | approximate: True  | maxdiff: 0.0
counts_sum      | exact: True  | approximate: True  | maxdiff: 0.0
counts          | exact: True  | approximate: True  | maxdiff: 0.0
norm_logits     | exact: False | approximate: True  | maxdiff: 8.381903171539307e-09
logit_maxes     | exact: True  | approximate: True  | maxdiff: 0.0
logits          | exact: True  | approximate: True  | maxdiff: 0.0
h               | exact: True  | approximate: True  | maxdiff: 0.0
W2              | exact: True  | approximate: True  | maxdiff: 0.0
b2              | exact: True  | approximate: True  | maxdiff: 0.0
hpreact         | exact: True  | approximate: True  | maxdiff: 0.0
bngain          | exact: True  | approximate: True  | maxdiff: 0.0
bnbias          | exact: True  | approximate: True  | maxdiff: 0.0
bnraw           | exact: True  | approximate: True  | maxdiff: 0.0
bnvar_inv       | exact: True  | approximate: True  | maxdiff: 0.0
bnvar           | exact: True  | approximate: True  | maxdiff: 0.0
bndiff2         | exact: True  | approximate: True  | maxdiff: 0.0
bndiff          | exact: False | approximate: False | maxdiff: 0.0010883426293730736
bnmeani         | exact: True  | approximate: True  | maxdiff: 0.0
hprebn          | exact: True  | approximate: True  | maxdiff: 0.0
embcat          | exact: True  | approximate: True  | maxdiff: 0.0
W1              | exact: True  | approximate: True  | maxdiff: 0.0
b1              | exact: True  | approximate: True  | maxdiff: 0.0
emb             | exact: True  | approximate: True  | maxdiff: 0.0
C               | exact: False | approximate: False | maxdiff: 0.04050402343273163
"""

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n
dprobs = (1.0 / probs) * dlogprobs  #this part here shows that if an element has a very low probability then its gradient would get boosted

dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
dcounts = counts_sum_inv * dprobs
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
dcounts += torch.ones_like(counts) * dcounts_sum
dnorm_logits = counts * dcounts

#norm_logits = logits - logit_maxes 
# the substraction means that broadcasting is happening 
#norm_logits.shape is [32, 27] and logits.shape is [32, 27] 
#while logit_maxes is [32, 1] so broadcasting is creating a copy to get 27 columns
#for dlogit_maxes we want to shape it back to [32, 1] so summing over the columns dimension
#NOTE: WHY - logit_maxes, this is to guarantee that the highest number that logits can have is zero
#this will bound counts = norm_logits.exp() and avoid exploding it 
dlogits = dnorm_logits.clone() #for safety use cloning
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True) 
#the dlogit_maxes values are extremely slow, close to zero
#this is telling us that logit_maxes is not impacting the loss function 
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes #an array with 1 at the index of the max value of logits
dh = dlogits @ W2.T
dW2 = h.T @ dlogits 
db2 = dlogits.sum(0) #no keepdim here because its shape is [27] and not [27, 1]
dhpreact = (1.0 - h**2) * dh
dbngain = (bnraw * dhpreact).sum(0, keepdim = True) #keepdim = True and summing row dimension to get [1, 64]
dbnraw = bngain * dhpreact
dbnbias = dhpreact.sum(0, keepdim = True)
dbndiff = bnvar_inv * dbnraw
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim = True)
dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2)*dbnvar
dbndiff += (2*bndiff)* dbndiff2
dhprebn = dbndiff.clone()
dbnmeani = - dbndiff.sum(0)
dhprebn += (1.0/n) * (torch.ones_like(hprebn)* dbnmeani)
dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0) 
demb = dembcat.view(emb.shape)
dC = torch.zeros_like(C)
for i in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[i, j]
        dC[ix] += demb[i, j]

cmp('logprobs', dlogprobs, logprobs)
cmp('probs', dprobs, probs)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
cmp('counts_sum', dcounts_sum, counts_sum)
cmp('counts', dcounts, counts)
cmp('norm_logits', dnorm_logits, norm_logits)
cmp('logit_maxes', dlogit_maxes, logit_maxes)
cmp('logits', dlogits, logits)
cmp('h', dh, h)
cmp('W2', dW2, W2)
cmp('b2', db2, b2)
cmp('hpreact', dhpreact, hpreact)
cmp('bngain', dbngain, bngain)
cmp('bnbias', dbnbias, bnbias)
cmp('bnraw', dbnraw, bnraw)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
cmp('bnvar', dbnvar, bnvar)
cmp('bndiff2', dbndiff2, bndiff2)
cmp('bndiff', dbndiff, bndiff)
cmp('bnmeani', dbnmeani, bnmeani)
cmp('hprebn', dhprebn, hprebn)
cmp('embcat', dembcat, embcat)
cmp('W1', dW1, W1)
cmp('b1', db1, b1)
cmp('emb', demb, emb)
cmp('C', dC, C)


# Exercise 2: backprop through cross_entropy but all in one go
# to complete this challenge look at the mathematical expression of the loss,
# take the derivative, simplify the expression, and just write it out

# forward pass

# before:
# logit_maxes = logits.max(1, keepdim=True).values
# norm_logits = logits - logit_maxes # subtract max for numerical stability
# counts = norm_logits.exp()
# counts_sum = counts.sum(1, keepdims=True)
# counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
# probs = counts * counts_sum_inv
# logprobs = probs.log()
# loss = -logprobs[range(n), Yb].mean()

# now:
loss_fast = F.cross_entropy(logits, Yb)
print(loss_fast.item(), 'diff:', (loss_fast - loss).item())

#backward pass
""" 
My solution

#math on paper:
#probs = py = exp(logity)/sum(exp(logitk) over all k)
#dL/dlogitj = (dL/dprobs)*(dprobs/dlogitj)
#dL/dlogitj = -(1/probs)(dprobs/dlogitj)
#dprobs/dlogitj = (d(exp(logity))/dlogitj)(sum) - dsum(exp(logity))) / (sum^2)
#case 1 : j = y
#dprobs/dlogitj = py( 1- py) 
#dL/dlogitj = py - 1 
#case 2 : j not y
#dprobs/dlogitj = - pjpy #since d(exp(logity))/dlogitj = 0 
#dL/dlogitj = pj
#so dL/dlogitj = pj - Yb[j] #since Yb[j] is 0 if y is not j and is 1 if y = j
#finally since loss = 1/n(the summation of - log(pj)) add the 1/n factor


yb = F.one_hot(Yb, num_classes = probs.shape[1])
dlogits =  (1.0/n)*(probs - yb)
cmp('logits', dlogits, logits)
"""

"""
My result 
logits          | exact: False | approximate: True  | maxdiff: 5.3551048040390015e-09
"""

#the loss for a batch is the average loss of all examples 
dlogits = F.softmax(logits, 1) 
dlogits[range(n), Yb] -= 1
dlogits /= n
cmp('logits', dlogits, logits)
#logits          | exact: False | approximate: True  | maxdiff: 6.51925802230835e-09
#dlogits make the system pull up (+ve) on the probability of the correct index and pull down on the wrong ones 
plt.figure(figsize=(4, 4))
plt.imshow(dlogits.detach(), cmap='gray')
#plt.show()

# Exercise 3: backprop through batchnorm but all in one go
# to complete this challenge look at the mathematical expression of the output of batchnorm,
# take the derivative w.r.t. its input, simplify the expression, and just write it out

# forward pass

# before:
# bnmeani = 1/n*hprebn.sum(0, keepdim=True)
# bndiff = hprebn - bnmeani
# bndiff2 = bndiff**2
# bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
# bnvar_inv = (bnvar + 1e-5)**-0.5
# bnraw = bndiff * bnvar_inv
# hpreact = bngain * bnraw + bnbias

# now:
hpreact_fast = bngain * (hprebn - hprebn.mean(0, keepdim=True)) / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True) + 1e-5) + bnbias
print('max diff:', (hpreact_fast - hpreact).abs().max())
# backward pass

# before we had:
# dbnraw = bngain * dhpreact
# dbndiff = bnvar_inv * dbnraw
# dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
# dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
# dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * dbnvar
# dbndiff += (2*bndiff) * dbndiff2
# dhprebn = dbndiff.clone()
# dbnmeani = (-dbndiff).sum(0)
# dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)

# calculate dhprebn given dhpreact (i.e. backprop through the batchnorm)
# (you'll also need to use some of the variables from the forward pass up above)
#dbnvar = (2.0/(n-1)) *(hprebn - bnmeani - (1.0/n)*(hprebn - 1)).sum(0)
#dxhat = (1.0/bnvar)*((bnvar**0.5)*(1.0 - 1.0/n) + 2.0/n*(dbnvar)*(bnvar + bnmeani))
#dhprebn = dhpreact* (bngain*dxhat) 
dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
cmp('hprebn', dhprebn, hprebn) 