import torch
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
print(len(words))
print(min(len(w) for w in words))

#counting bigrams
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

sorted(b.items(), key = lambda kv: - kv[1])

"""
chars = sorted(list(set(''.join(words))))
stoi = {s:i for i,s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27
N = torch.zeros((28, 28), dtype = torch.int32)
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
"""

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
N = torch.zeros((27, 27), dtype = torch.int32)
itos = {i:s for s,i in stoi.items()}
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

#Plotting the heat chart
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');
#plt.show()

"""
p = N[0].float() #probabiliyy distribution
p = p/p.sum() #normalization so that they all add up to 1 

#Generator on pytorch makes things deterministic so i get same number each time
g = torch.Generator().manual_seed(2147483647)
p1 = torch.rand(3, generator=g)
p1 = p1/p1.sum()

#you can specify the distribution of the samples (here they follow the probability distribution of p)
#if p[0] = 0.66 then 66% of the samples would be 0
torch.multinomial(p, num_samples=20, replacement=True, generator=g)
"""
#P= N.float()
P = (N+1).float() #adding +1 here for smoothness so nothing is 0 probability so the loss would not be -inf
P /= P.sum(dim=1, keepdim=True) #keepdim = True so that the output tensor has size ([27, 1]) so 27 rows 1 col instead of ([27])
#note 1D tensor has no concept of grid rows or columns
#how tensors divide, first must follow broadcasting rule 
#Two tensors are broadcastable if you align their shapes starting from the trailing (rightmost) dimension and work your way backward. For each dimension, the sizes must satisfy one of these conditions:
# 1.   They are equal.
# 2. One of them is exactly 1.
# 3. One of them does not exist (meaning one tensor has fewer dimensions than the other
# so this works for 
# 27, 27
# 27 , 1
# then pytorch would automatically expand the smaller tensor to match the larger ones for the element-wise operations
# so here ([27, 1]) would be copied and expanded to fill ([27, 27])
#but be careful if we had ([27]) then right to left 
# 27, 27
#.  , 27 so what happens here this will be expanded to 1, 27 making it a row vector which is not what we want 
#one more note using "/=" to have in place memory makes it faster

g = torch.Generator().manual_seed(2147483647)
for i in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        #p = N[ix].float()
        #p = p/p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    #print(''.join(out))


#Evaluate model:

#probably the bigram model assigned for the biagram 
# GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

log_likelihood = 0.0
n = 0
for w in words[:3]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        #print(f'{ch1}{ch2}: {prob:.4f}')
nll = -log_likelihood
#print(f'{log_likelihood=}')
#print(f'{nll=}')
#normalizing it 
#print(f'{nll/n=}')

#Build neural net for the Bigram problem
#Create the training set of bigrams(x,y)
xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

import torch.nn.functional as F

xenc = F.one_hot(xs, num_classes = 27).float()
#one hot is the function that will allow us to have the inputs as the vector [0, 0 , 1] for example to represent the 3rd item of 3 classes
#casting it to .float() is important step because the NN needs the input to be in float for the rest of the math to take place
W = torch.randn(27,1)
print(f'{xenc.shape=}')
print(xenc @ W)
# @ operation is the dot product, it does follow broadcasting rules but a little different
# example: (5, 27) @ (27, 27) -> (5, 27)
# (M, K) @ (K, N) --> (M, N)
# (B, M, K) @ (K, N) --> (B, M, K) @ (1, K, N) --> (B, M, N)