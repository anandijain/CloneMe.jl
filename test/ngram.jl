using CloneMe
# Read and preprocess text
s = read("C:/Users/anand/src/transcript_grabber/data/dataset.txt", String)
na = filter(!isascii, s)
t = filter(isascii, s)
l = length(t)
cs = collect(t)
chars = sort(unique(t))
nc = length(chars)
n = 4

stoi, itos = ix_maps(chars)

getd(d, xs) = map(x -> d[x], xs)
enc(x) = getd(stoi, x)
dec(x) = getd(itos, x)

split_idx = round(Int, 0.9 * l)
(X, Y) = CloneMe.get_batch(t, 1:(split_idx-n), n-1)
(Xv, Yv) = CloneMe.get_batch(t, (split_idx-n):(l-n), n-1)

M = counts_tensor(t, n, enc);
# smooth
# M .+= 1
P = M ./ sum(M, dims=n);

maxlen = 100000
start = " th"
gen = collect(start)
for i in 1:maxlen
    context = gen[end-(n-1)+1:end]
    idxs = enc(context)
    probs = vec(P[idxs..., :])
    next_char = itos[sample(Weights(probs))]
    push!(gen, next_char)
end
println(String(gen))

# todo come up with a fast loss calculation on test set 