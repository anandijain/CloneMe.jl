module CloneMe
using Flux
using StatsBase
function char_ps(chars)
    # we don't sort intentionally
    # we want the stop char to be idx 1
    # chars = sort(unique(chars))
    @assert allunique(chars)
    chars .=> eachindex(chars)
end

function ix_maps(chars)
    stoi_ps = char_ps(chars)
    stoi = Dict(stoi_ps)
    itos = Dict(reverse.(collect(stoi)))
    stoi, itos
end

function alphabet(t; pad_char=';')
    cs = sort(unique(t))
    @assert pad_char ∉ cs
    [pad_char, cs...]
end

function get_example(t, i, block_size)
    x = t[i:i+block_size-1]
    y = t[i+block_size]
    x, y
end

unzip(xs) = first.(xs), last.(xs)

function get_batch(t, is, block_size)
    xs, ys = unzip(get_example.((t,), is, (block_size,)))
    (stack(xs), ys)
end

function generate(model, n, block_size, xenc, itos; maxlen=100)
    outs = []
    for _ in 1:n
        # xenc = ones(Int, block_size) # ";;;"
        # xenc = rand(1:nc, block_size)
        
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
        push!(outs, o)
    end
    outs
end

# random stuff 

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

export char_ps, ix_maps, alphabet, get_example, get_batch, generate

end # module CloneMe
