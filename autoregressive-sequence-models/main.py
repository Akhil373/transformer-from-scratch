# %%
from numpy.random import Generator
from os import login_tty
import matplotlib
import matplotlib.pyplot as plt
import torch
import os
os.chdir("/home/axle/code/ML/transformer-from-scratch/autoregressive-sequence-models/")
# %%
words = open("./names.txt", "r").read().splitlines()
words[:10]
min(len(w) for w in words)
max(len(w) for w in words)
# %%
b={}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
        # print(bigram)
# %%
sorted(b.items(), key=lambda kv: -kv[1])
# %%
N = torch.zeros((27,27), dtype=torch.int32)
N
# %%
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
itos
# %%
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
# %%
%matplotlib inline

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i ,N[i,j].item(), ha='center', va='top', color='gray')
plt.axis('off')
plt.savefig('output.png')
# %%
N[0, :]

# %%
p = N[0].float()
p = p / p.sum()
print(p)
p.sum().item()
# %%
g = torch.Generator().manual_seed(2147483647)
idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
itos[idx]
# %%
g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p/ p.sum()
p

# %%
r = torch.multinomial(p, num_samples=100, replacement=True, generator=g)
print(r)
torch.bincount(r)
# %%
P = N.float()
P /= P.sum(1, keepdim=True)
P[0].sum()
# 27, 27
# 27, 1
# %%
g = torch.Generator().manual_seed(2147483647)
for _ in range(5):
    out = []
    idx = 0
    while True:
        p = P[idx]
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[idx])
        if idx == 0:
            break
    print(''.join(out))
# %%
log_likelihood = 0.0
n = 0
for w in words[:3]:
    chs = ['.'] + list(w) +['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
print(log_likelihood)
nll = -log_likelihood
print(f"negative log likelihood: {(nll/n).item()}")


# %% neural network to train the bigram
# creating training dataset
xs,  ys= [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        xs.append(idx1)
        ys.append(idx2) # we are predicting the next char given first char

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print("number of examples: ", num)

# init network
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad=True)
# %%
# gradient descent
for k in range(201):
    # forward pass
    import torch.nn.functional as F
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = (xenc @ W) # log counts
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean()
    prev_loss = loss
    if k%50==0:
        print(f"step: {k}, loss: {loss.item()}")
    # backward pass
    W.grad = None
    loss.backward()
    # update step
    W.data += -50 * W.grad
# %%
g = torch.Generator().manual_seed(2147483647)
for _ in range(5):
    out = []
    idx = 0
    while True:
        xenc = F.one_hot(torch.tensor([idx]), num_classes=27).float()

        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)

        idx = torch.multinomial(probs[0], num_samples=1, replacement=True, generator=g).item()
        
        if idx == 0:
            break
        
        out.append(itos[idx])
    print(''.join(out))
# %%
'''
## counting
Out[15]: ✓ Done 0.0s
cexze.
momasurailezitynn.
konimittain.
llayn.
ka.

## nn model
Out[98]: ✓ Done 0.0s
cexze
momasurailezitynn
konimittain
llayn
ka
'''
