# this file walks through some of the math behind a basic MLP language model

using Flux
using Flux: logitcrossentropy, onehotbatch
using CloneMe
using Plots 

s = read("C:/Users/anand/src/transcript_grabber/data/dataset.txt", String)
na = filter(!isascii, s)

t = filter(isascii, s)
l = length(t)

# take 3 characters and predict the fourth 
block_size = 3
split_idx = round(Int, 0.9 * l)

chars = alphabet(t)
nc = length(chars)
stoi, itos = ix_maps(chars)

enc(x) = getd(stoi, x)
dec(x) = getd(itos, x)

ts = collect(t)
enct = enc(ts)

get_example(i) = CloneMe.get_example(enct, i, block_size)
get_batch(is) = CloneMe.get_batch(enct, is, block_size)

(x1, y1) = get_example(1)
x1d, y1d = dec.((x1, y1))
@assert (x1d, y1d) == (['y', 'o', 'u'], '\n')

T = block_size
C = n_embd = 2
nh = 10

emb = Embedding(nc, n_embd)
reshape_fn = x -> reshape(x, (n_embd * block_size, :))
ex1 = emb(x1) # 2x3
rx1 = reshape_fn(ex1) # 6x1
ew = emb.weight
l1 = Dense(n_embd * block_size, nh, tanh)
l1x1 = l1(rx1)
l2 = Dense(nh, nc)

l2x1 = l2(l1x1)
model = Chain(
    emb,
    x -> reshape(x, (n_embd * block_size, :)),
    l1,
    l2
)

@assert l1x1 == model[1:3](x1)
logits = model(x1)
# exponentiation is done just to get the numbers positive and retain the order 
# its just the first step of making the output a true probability 
counts = exp.(logits) 
histogram(logits)
probs = counts ./ sum(counts)

# the predicted probability of the true next characters
pred_gt = probs[y1]
loss = -mean(log(pred_gt))
softmax(logits) ≈ counts ./ sum(counts)
# we take the -log because if the model predicts prob 1, then the loss is zero log(1)==0
# and the loss is Inf if pred prob 0  

y1oh = onehotbatch(y1, 1:nc)
loss2 = logitcrossentropy(logits, y1oh)
@assert loss ≈ loss2

lsm = logsoftmax(logits;dims=1)
y1oh .* lsm
mean(.-sum(y1oh .* logsoftmax(logits; dims=1); dims=1))

# you might be able to save a few calls to `log` by only computing the log of the actual ys


# now a fwd pass with a batch
(xb, yb) = get_batch(1:2)
logits = model(xb) # 84x2
counts = exp.(logits)

# each column of logits is a separate prediction, so you want to sum along dimension 1
# thats the dimension of length 84 in order to correctly make probabilities from  the logits
probs = counts ./ sum(counts, dims=1)
@assert sum(probs[1:end, 1]) ≈ 1

# here we calculate the expected loss with uniform probabilities, which are to be sort of expected when randomly init weights
-log(1/nc)

batch_size = 64
(X, Y) = get_batch(1:(split_idx - block_size))
loader = Flux.DataLoader((data=X, label=Y), batchsize=batch_size, shuffle=false);
x1l, y1l = first(loader)

lr = 1e-2
opt_state = Flux.setup(Flux.Descent(lr), model)
model
for (i, (x, y)) in enumerate(loader)
    loss, grads = Flux.withgradient(model) do m
        logits = m(x)
        loss = logitcrossentropy(logits, onehotbatch(y, 1:nc))
    end
    @info i loss
    if i == 1000 
        break
    end
    Flux.update!(opt_state, model, grads[1])
end