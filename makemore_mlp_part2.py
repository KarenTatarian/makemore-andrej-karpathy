import torch 
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt 

#following the paper Bengio et al. 2003 MLP language model paper (pdf): https://www.jmlr.org/papers/volume3/b... 

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
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

g = torch.Generator().manual_seed(2147483647)

#Embeddings 
#C = torch.rand(27, 2, generator = g)
#so each character will have a 2D embedding 
#increased embedding dimensions because the model was underfitting so we need to increase complexity of model 
C = torch.rand(27, 10, generator = g)


#emb = C[X] 
#------------
#emb.shape is torch.Size([32, 3, 2]) 
#32 is all the examples we have and 3 is the context size so the three consecutive characters and 2 is the embedding dimension of each of these characters
#The way pytorch stores tensors is like a stack of all the elements inside the tensor and then stores the shape and stride
#this allows pytorch to be able to manipulate how the shape of tensor by re-arranging the elements in order in any shape we want 
#print(emb[0])
#print(emb.view(32,6)[0])
#emb.view(32, 6) is needed to able to perform dot product on 6, 100
# ----- .VIEW() PYTORCH -----
#emb.view(32, 6) explicitly forces the tensor into exactly 32 rows and 6 columns
#emb.view(-1, 6) tells PyTorch: "Give me exactly 6 columns, and you do the math to figure out how many rows I need." 
#Because 32 × 3 × 2 = 192 total elements, and 192 / 6 = 32, PyTorch automatically infers the row dimension to be 32.

#-------constructing the first hidden layer 
#W1 = torch.rand(6, 300, generator = g) #6 because 3x2 input where 3 is the context size and 2 is each's embedding dimensions 
#100 chosen at random, how many neurons we want, how big the layer is 
#b1 = torch.rand(300, generator = g)
W1 = torch.rand(30, 200, generator = g)
b1 = torch.rand(200, generator = g)

#h = torch.tanh(emb.view(-1, 6) @ W1 + b1) #with broadcasting this will work 
# 32, 100
#    , 100
#so the above follow the broadcasting rules h.shape is torch.Size([32, 100])

#-----constructing second hidden layer 
#W2 = torch.rand(300, 27, generator = g)
b2 = torch.rand(27, generator = g)
W2 = torch.rand(200, 27, generator = g)

parameters = [C, W1, b1, W2, b2]

#logits = h @ W2 + b2
#counts = logits.exp()
#prob = counts/counts.sum(dim = 1, keepdims = True)
#loss = - prob[torch.arange(X.shape[0]), Y].log().mean() + 0.05*(W2**2).mean()
#instead of calculating the loss manually will use cross_entropy which numerically handles edge cases better and uses one kernel

#to decide optimal learning rate 
lre = torch.linspace(-3, 0, 1000)
#It creates a tensor containing evenly spaced numbers from start to end, including both endpoints
lrs = 10**lre 

for p in parameters:
    p.requires_grad = True

lr = []
losses = []
steps = []
lri = 10**-1
max_steps = 200000

for i in range(max_steps):
    #minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))

    #forward pass
    emb = C[Xtr[ix]]
    #h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    l2_reg = sum((w**2).sum() for w in [C, W1, W2])
    total_elements = sum(w.nelement() for w in [C, W1, W2])
    #!!!! Note probs = F.softmax(logits, dim = 1) is not needed because F.cross_entropy expects logits 
    #it does softmax internally
    data_loss = F.cross_entropy(logits, Ytr[ix])
    l2_penalty =  + 0.5*l2_reg/total_elements 
    loss = data_loss + l2_penalty
    #print(loss.item())

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #lri = lrs[i]
    #lri = 0.1 if i < 10000 else 0.01
    #lri = 0.99*lri --> too aggressive 

    if i % 10000 == 0:
        lri = 0.99*lri
        print(f'{i:7d}/{max_steps:7d}: {loss.item(): .4f}')
    #update parameters
    for p in parameters:
        p.data += -lri*p.grad 
    
    losses.append(loss.log10().item())
    lr.append(lri)
    steps.append(i)

print(losses[len(losses) - 1])
#plotting shows us where is the ideal area 
plt.plot(steps, losses)
plt.show()
#can also plot and check lre[i] so the exponent of the learning it to get more precise 
#in this we see the optimal exponent lr is 10^-1 which is 0.1

#note: went from underfitting to overfitting 

#Evaluating the training 
emb = C[Xdev]
#h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(loss)

# visualize dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
plt.grid('minor')
plt.show()

#testing some word generations 
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0]*block_size 
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim = 1)
        ix = torch.multinomial(probs, num_samples = 1, replacement = True, generator = g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))

