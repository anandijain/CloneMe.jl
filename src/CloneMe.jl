module CloneMe

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

export char_ps, ix_maps, alphabet, get_example, get_batch

end # module CloneMe
