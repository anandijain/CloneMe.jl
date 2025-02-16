# Read and preprocess text
s = read("C:/Users/anand/src/transcript_grabber/data/dataset.txt", String)
na = filter(!isascii, s)
t = filter(isascii, s)
l = length(t)
cs = collect(t)
chars = sort(unique(t))
nc = length(chars)

# Create encoding/decoding maps
stoi, itos = ix_maps(chars)

getd(d, xs) = map(x -> d[x], xs)
enc(x) = getd(stoi, x)
dec(x) = getd(itos, x)

# (Assuming block_size is defined somewhere, and your batch function works similarly)
split_idx = round(Int, 0.9 * l)
(X, Y) = CloneMe.get_batch(t, 1:(split_idx-block_size), 1)
(Xv, Yv) = CloneMe.get_batch(t, (split_idx-block_size):(l-block_size), 1)
# Build a 3-tensor for trigram counts, with workaround for getd returning a 0-dim array.
function counts_tensor_trigram(str)
    cs = collect(str)
    nc = length(chars)  # assuming `chars` (sorted unique characters) is defined globally
    # Initialize a tensor of shape (nc, nc, nc)
    M = zeros(Int, nc, nc, nc)
    # Loop over triples of consecutive characters
    for (c1, c2, c3) in zip(cs, cs[2:end], cs[3:end])
        i = enc(c1)[]   # Extract scalar from 0-dim array
        j = enc(c2)[]
        k = enc(c3)[]
        M[i, j, k] += 1
    end
    M
end

# Compute the trigram counts tensor
M = counts_tensor_trigram(t)

# (Optional) Add Laplace smoothing if desired
# M .+= 1

# Convert counts to probabilities: for each context (i, j), normalize over the third dimension.
P = M ./ sum(M, dims=3)

# Generation example using the probability tensor:
maxlen = 100
c1 = 'e'   # first seed character
c2 = 'x'   # second seed character (choose as desired)
gen = [c1, c2]
for i in 1:maxlen
    # Extract scalar indices for the current context
    i1 = enc(c1)[]
    i2 = enc(c2)[]
    # Get the probability vector for the current context
    probs = vec(P[i1, i2, :])
    # Sample the next character using the probability distribution
    c3 = itos[sample(Weights(probs))]
    push!(gen, c3)
    # Update context: shift by one character
    c1, c2 = c2, c3
end

println(String(gen))

# Build context-target pairs for evaluation
# For each valid index i, we treat the context as (t[i], t[i+1]) and the target as t[i+2]
X_trigram = [(enc(t[i])[], enc(t[i+1])[]) for i in 1:(l-2)]
Y_trigram = [enc(t[i+2])[] for i in 1:(l-2)]

# For each context pair, extract the probability vector from the 3-tensor P.
# P is defined such that P[i, j, :] is the probability distribution over next characters
# given that the current context is (i, j).
predictions = hcat([vec(P[i, j, :]) for (i, j) in X_trigram]...)
# The resulting `predictions` has shape (nc, num_examples)

# Convert the targets into one-hot vectors.
targets = onehotbatch(Y_trigram, 1:nc)

# Compute the test loss using your logitcrossentropy function.
test_loss = logitcrossentropy(predictions, targets)

println("Test loss: ", test_loss)

# with 1 smoothing
# julia > test_loss = logitcrossentropy(predictions, targets)
# 4.127769320649866

# no smoothing 
# julia > test_loss = logitcrossentropy(predictions, targets)
# 4.123863716473766