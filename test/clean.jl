using Flux
using Flux: logitcrossentropy, onehotbatch
using CloneMe
using Plots

s = read("C:/Users/anand/src/transcript_grabber/data/dataset.txt", String)
na = filter(!isascii, s)

t = filter(isascii, s)
l = length(t)

# take 3 characters and predict the fourth 
block_size = 8
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

(X, Y) = get_batch(1:(split_idx-block_size))
(Xv, Yv) = get_batch((split_idx-block_size):l-block_size)

# model 
T = block_size
C = n_embd = 10
nh = 100

emb = Embedding(nc, n_embd)
reshape_fn = x -> reshape(x, (n_embd * block_size, :))
l1 = Dense(n_embd * block_size, nh, tanh)
l2 = Dense(nh, nc)

model = Chain(
    emb,
    x -> reshape(x, (n_embd * block_size, :)),
    l1,
    l2
)

# train 
batch_size = 64
loader = Flux.DataLoader((data=X, label=Y), batchsize=batch_size, shuffle=true)
loader2 = Flux.DataLoader((data=X, label=Y), batchsize=batch_size, shuffle=false)
x1l, y1l = first(loader2)

lr = 1e-2
opt_state = Flux.setup(Flux.Descent(lr), model)

losses = []
test_losses = []
for (i, (x, y)) in enumerate(loader)
    loss, grads = Flux.withgradient(model) do m
        logits = m(x)
        loss = logitcrossentropy(logits, onehotbatch(y, 1:nc))
    end
    push!(losses, loss)
    if i % 1000 == 0
        test_loss = logitcrossentropy(model(Xv), onehotbatch(Yv, 1:nc))
        push!(test_losses, (i, test_loss))
        @show i loss test_loss
    end
    # if i == 1000
    #     break
    # end
    Flux.update!(opt_state, model, grads[1])
end

p = plot(losses)
plot!(p, unzip(test_losses)...)

function generate(model, n, block_size; maxlen=100)
    outs = []
    for _ in 1:n
        # xenc = ones(Int, block_size) # ";;;"
        # xenc = rand(1:nc, block_size)
        xenc = enc(collect(";;;;;you"))
        out = []
        i = 1
        while true
            logits = model(reshape(xenc, (1, block_size)))
            counts = vec(softmax(logits))
            ix = sample(Weights(counts))
            push!(out, itos[ix])
            if ix == 1 || i == maxlen
                break
            end
            circshift!(xenc, -1)
            xenc[3] = ix
            i += 1
        end
        o = join(out)
        # println(o)
        push!(outs, o)
    end
    outs
end

outs = generate(model, 10, block_size)


# visualize embedding
function embedding_plot(model)
    C = model.layers[1].weight

    emb_xys = eachcol(C)
    xlims = extrema(C[1, :])
    ylims = extrema(C[1, :])
    uxys = unzip(emb_xys)

    pl = scatter()
    zuxt = collect(zip(uxys..., chars))
    annotate!(pl, zuxt)
    xlims!(pl, xlims)
    ylims!(pl, ylims)
    pl
end
embedding_plot(model)