from pandas.core.internals.blocks import ensure_block_shape
from numpy import block
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
os.chdir("/home/axle/code/ML/transformer-from-scratch/autoregressive-sequence-models/")
# %%
words = open("names.txt").read().splitlines()
print(words[:8])
print(len(words))
# %%
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
itos
# %%
def build_dataset(words):
    block_size =3
    X, Y =[],[]
    for w in  words:
        # print(w)
        context = [0] * block_size
        for ch in w+'.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '--->', ch)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X,Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
# %%
print(n1, n2)
# %%
g = torch.Generator().manual_seed(47)
C = torch.randn((27,10), generator=g)
W1 = torch.randn([30, 200], generator=g)
B1 = torch.randn([200], generator=g)
W2 = torch.randn([200, 27], generator=g)
B2 = torch.randn([27], generator=g)
parameters = [C, W1, B1, W2, B2]
# %%
Xtr.shape, Ytr.shape, Xtr.dtype, Ytr.dtype # dataset
# %%
sum(p.nelement() for p in parameters)
# %%
for p in parameters:
    p.requires_grad = True
# %%
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
# %%
lri = []
lossi = []
stepi = []
# %%
for i in range(50000):
# minibatch size 32
    ix = torch.randint(0, Xtr.shape[0], (32,))

# forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + B1) # shape = [32, 100]
    logits = h @ W2 + B2
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdims=True)
    # loss = -prob[torch.arange(32), Y].log().mean()
    loss = F.cross_entropy(logits, Ytr[ix])

# backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    # lr = lrs[i]
    lr = 0.001
    for p in parameters:
        p.data += -lr * p.grad
    # lri.append(lre[i])
    stepi.append(i)
    lossi.append(loss.log10().item())
print("loss:",loss.item())
# %%
# print("best learning rate:",lrs[lossi.index(min(lossi))].item())
plt.plot(stepi, lossi)
# plt.savefig('output.png')
# %%
emb = C[Xtr]
h = torch.tanh(emb.view(-1, 30) @ W1 + B1)
logits = h @ W2 + B2
loss = F.cross_entropy(logits, Ytr)
loss
# %%
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + B1)
logits = h @ W2 + B2
loss = F.cross_entropy(logits, Ydev)
loss
# %%
plt.figure(figsize=(8,8))
plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va='center', color='white')
plt.grid('minor')
# %%
g = torch.Generator(47+10)
block_size = 3
for _ in range(0):
    out = []
    context = [0] * block_size
    while True:
        emd = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + B1)
        logits = h @ W2 + B2
        probs = F.softmax(logits,dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append[ix]
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))
