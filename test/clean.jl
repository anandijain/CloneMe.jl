using Flux
using Flux: logitcrossentropy, onehotbatch
using CloneMe
using Plots
using BenchmarkTools
using CUDA, cuDNN
CUDA.allowscalar(false)
device = gpu_device()

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

model_cpu = Chain(
    emb,
    x -> reshape(x, (n_embd * block_size, :)),
    l1,
    l2
) 
model_gpu = deepcopy(model_cpu) |> device

model = model_gpu

# train 
batch_size = 64
loader = Flux.DataLoader((data=X, label=Y), batchsize=batch_size, shuffle=true)
loader2 = Flux.DataLoader((data=X, label=Y), batchsize=batch_size, shuffle=false)
xb, yb = first(loader2)

lr = 1e-2
opt_state = Flux.setup(Flux.Descent(lr), model)

losses = []
test_losses = []
for (i, xy_cpu) in enumerate(loader)
    x, y = xy_cpu |> device
    loss, grads = Flux.withgradient(model) do m
        logits = m(x)
        loss = logitcrossentropy(logits, onehotbatch(y, 1:nc))
    end
    Flux.update!(opt_state, model, grads[1])

    push!(losses, loss)
    if i % 1000 == 0
        test_loss = logitcrossentropy(model(Xv), onehotbatch(Yv, 1:nc))
        push!(test_losses, (i, test_loss))
        @show i loss test_loss
    end
    # if i == 10000
    #     break
    # end
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

@benchmark model(xb)
# without CUDA 


# julia> @benchmark model(xb)
# BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
#  Range (min … max):   9.700 μs … 168.900 μs  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     13.800 μs               ┊ GC (median):    0.00%
#   ▆▇▆██▇▆▅▄▄▄▃▃▂▂▁▁                       ▁ ▁▁▁                ▂
#   ████████████████████▇▇▆▇▅▅▃▅▄▅▄▅▅▆▆█▇████████████▇█▆▆▆▅▆▆▅▃▆ █
#   9.7 μs        Histogram: log(frequency) by time        63 μs <

#  Memory estimate: 25.19 KiB, allocs estimate: 15.

# julia> @benchmark model(xb)
# BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
#  Range (min … max):   9.500 μs … 196.200 μs  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     13.300 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   16.600 μs ±  11.929 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

#   █▆▇█▇▆▅▅▄▄▃▃▂▁▁▁                                             ▂
#   █████████████████▇█▇▆▇▆▅▅▅▃▅▆▆▇▇▇██████▇▇▇▆▇▆▇▇▅▆▆▅▄▆▄▆▅▅▅▄▅ █
#   9.5 μs        Histogram: log(frequency) by time      69.6 μs <

#  Memory estimate: 25.19 KiB, allocs estimate: 15.

xb_gpu = xb |> device
yb_gpu = yb |> device
@benchmark model(xb_gpu)
# with CUDA

# julia> @benchmark model(xb_gpu)
# BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
#  Range (min … max):  50.000 μs … 805.700 μs  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     70.900 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   91.059 μs ±  52.991 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

#    ▇█▇▇▇▆▅▅▄▄▃▃▃▂▂▂▂▂▂▂▁▁▁▁▂▂▂▂▂▂▁▂▁▁▁                         ▂
#   ▇████████████████████████████████████▇▇▇▆▇▆▆▇▆▇▇▆▆▆▆▄▆▆▅▆▆▄▅ █
#   50 μs         Histogram: log(frequency) by time       295 μs <

#  Memory estimate: 9.86 KiB, allocs estimate: 347.

# its like 5 times slower with CUDA



function update_loss(model, x, y)
    opt_state = Flux.setup(Flux.Descent(1e-2), model)
    loss, grads = Flux.withgradient(model) do m
        logits = m(x)
        loss = logitcrossentropy(logits, onehotbatch(y, 1:nc))
    end
    Flux.update!(opt_state, model, grads[1])
end

@benchmark update_loss(model_cpu, xb, yb)
# julia> @benchmark update_loss(model_cpu, xb, yb)
# BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
#  Range (min … max):  215.800 μs … 548.927 ms  ┊ GC (min … max):  0.00% … 99.83%
#  Time  (median):     272.000 μs               ┊ GC (median):     0.00%
#  Time  (mean ± σ):   397.079 μs ±   5.720 ms  ┊ GC (mean ± σ):  25.36% ±  5.51%

#     ▂▅▇█▃          ▁
#   ▂▅█████▇▅▄▃▃▂▂▃▅███▇▆▅▄▃▃▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▃
#   216 μs           Histogram: frequency by time          575 μs <

#  Memory estimate: 364.89 KiB, allocs estimate: 361.


@benchmark update_loss(model_gpu, xb_gpu, yb_gpu)
# julia> @benchmark update_loss(model_gpu, xb_gpu, yb_gpu)
# BenchmarkTools.Trial: 3363 samples with 1 evaluation per sample.
#  Range (min … max):  835.700 μs …   4.040 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):       1.409 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):     1.475 ms ± 341.859 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

#          ▁▂▃▆▄▇▆▇▇▆▇▇▇█▆▅▂▅▅▃▃
#   ▁▂▂▃▄▄▆████████████████████████▆█▇▇▇▆▆▆▆▄▄▄▄▃▄▃▂▃▃▂▂▂▂▂▂▁▂▁▁▂ ▅
#   836 μs           Histogram: frequency by time         2.52 ms <

#  Memory estimate: 96.29 KiB, allocs estimate: 2570.


CUDA.@profile model_gpu(xb_gpu)
@benchmark CUDA.@sync model_gpu($xb_gpu)