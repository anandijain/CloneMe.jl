s = read("C:/Users/anand/src/transcript_grabber/data/dataset.txt", String)
na = filter(!isascii, s)
t = filter(isascii, s)
l = length(t)
cs = collect(t)
chars = sort(unique(str))
nc = length(chars)

stoi, itos = ix_maps(chars)
enc(x) = getd(stoi, x)
dec(x) = getd(itos, x)

split_idx = round(Int, 0.9 * l)
(X, Y) = CloneMe.get_batch(t, 1:(split_idx-block_size), 1)
(Xv, Yv) = CloneMe.get_batch(t, (split_idx-block_size):(l-block_size), 1)

"""
buids a matrix given a string for the transitions

"""
function counts_table(str)
    chars = sort(unique(str))
    nc = length(chars)
    cs = collect(str)
    tly = tally(zip(cs, cs[2:end]))
    enc_tly = map(enc ∘ first, tly) .=> last.(tly)
    M = zeros(Int, (nc, nc))
    for (x, y) in enc_tly
        M[x...] = y
    end
    M
end


M = counts_table(t)

# smoothing (optional)
# M .+= 1

P = M ./ sum(M;dims=2)
weight_vecs = map(Weights ∘ vec, eachrow(P))
# 'e'th row. and sample from those weights 
plot(weight_vecs[stoi['e']])
sample(weight_vecs[stoi[c]])
maxlen = 100
c = 'e'
gen = [c]
for i in Base.OneTo(maxlen)
    c = itos[sample(weight_vecs[stoi[c]])]
    push!(gen, c)
end
String(gen)

bigram(x) = sample(weight_vecs[x])

# evaluate the test loss with bigram model 
Xve, Yve = enc(Xv), enc(Yv)

test_loss = logitcrossentropy(P[Xve, :][1, :, :]', onehotbatch(Yve, 1:nc))