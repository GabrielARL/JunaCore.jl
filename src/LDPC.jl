module LDPC

# Thin wrapper around Radford Neal's LDPC tools (make-ldpc / make-gen / print-pchk),
# https://glizen.com/radfordneal/ftp/LDPC-2012-02-11/index.html
#
# A code is built with:
#     make-ldpc pchk  M  N  seed  <method...>
# where <method...> is e.g. `evenboth 3 no4cycle` or `evencol 3 no4cycle`. The
# number (here 3) is the per-column check count `dc` -- how many checks each bit
# participates in. It may also be a degree-distribution string understood by
# make-ldpc, e.g. "0.4x2/0.6x3". `build` keeps that fully general.

export create, build, generator, read_H

method_args(method, dc; no4cycle=true) =
  no4cycle ? [string(method), string(dc), "no4cycle"] : [string(method), string(dc)]

_ok(p) = isfile(p) && filesize(p) > 0

"""
    build(k, n; method="evenboth", dc=3, no4cycle=true, seed=1, dir=<cache>)

Construct an `(n-k) x n` LDPC code and return `(; icols, gen, H, pchk)`:
`icols`/`gen` from the systematic generator, and the parity-check matrix `H`.

`method` is `"evenboth"` or `"evencol"`; `dc` is the per-column check count
(an `Int`, or a make-ldpc degree-distribution string). Artifacts are cached in
`dir`, keyed by the FULL spec (k, n, seed, method, dc, no4cycle), so changing any
knob produces a fresh code instead of silently reusing a stale file.
"""
function build(k, n; method="evenboth", dc=3, no4cycle=true, seed=1,
               dir=joinpath(tempdir(), "jldpc_cache"))
  mkpath(dir)
  margs = method_args(method, dc; no4cycle=no4cycle)
  tag = replace(join([k, n, seed, margs...], "_"), r"[^A-Za-z0-9_]" => "")
  base = joinpath(dir, "ldpc-$(tag)")
  pchk, gen, htxt = base * ".pchk", base * ".gen", base * ".H"

  if !_ok(pchk)
    _run(Cmd(vcat([_tool("make-ldpc"), pchk, string(n - k), string(n), string(seed)], margs)))
  end
  if !_ok(gen)
    _run(Cmd([_tool("make-gen"), pchk, gen, "dense"]))
  end
  if !_ok(htxt)
    open(htxt, "w") do io
      run(pipeline(`$(_tool("print-pchk")) $pchk`; stdout=io, stderr=devnull))
    end
  end

  icols, G = generator(gen)
  (; icols, gen=G, H=read_H(htxt, n - k, n), pchk)
end

"""
    create(k, n, opts="1 evenboth 3 no4cycle")

Backward-compatible entry: `opts` is the raw `"seed method dc [no4cycle]"` string
passed to make-ldpc. Returns `(icols, gen)` like before.
"""
function create(k, n, opts="1 evenboth 3 no4cycle")
  t = split(opts)
  length(t) >= 3 || throw(ArgumentError("opts must be \"seed method dc [no4cycle]\", got \"$opts\""))
  r = build(k, n; seed=parse(Int, t[1]), method=t[2], dc=t[3],
            no4cycle=(length(t) >= 4 && t[4] == "no4cycle"))
  (r.icols, r.gen)
end

"""
    read_H(filename, m, n) -> BitMatrix

Parse the `print-pchk` text dump of an `m x n` parity-check matrix.
"""
function read_H(filename, m, n)
  H = falses(m, n)
  for line in eachline(filename)
    mm = match(r"^ *(\d+):(.+)$", line)
    mm === nothing && continue
    row = parse(Int, mm[1]) + 1
    for col0 in parse.(Int, split(mm[2]))
      H[row, col0 + 1] = true
    end
  end
  H
end

function generator(filename)
  open(filename) do io
    read(io, UInt32) == 0x00004780 || throw(ErrorException("Bad generator: magic number mismatch - $(filename)"))
    read(io, UInt8) == 0x64 || throw(ErrorException("Bad generator: must be dense - $(filename)"))
    p = Int(read(io, UInt32))
    n = Int(read(io, UInt32))
    icols = [Int(read(io, UInt32)) + 1 for _ ∈ 1:n] |> invperm
    Int(read(io, UInt32)) == p || throw(ErrorException("Bad generator: row size mismatch - $(filename)"))
    Int(read(io, UInt32)) == n - p || throw(ErrorException("Bad generator: column size mismatch - $(filename)"))
    G = zeros(Bool, p, n-p)
    try
      v = Vector{UInt8}(undef, 4*ceil(Int, p/32))
      setval!(G, i, j, x) = i ≤ size(G,1) && (G[i,j] = x)
      for i ∈ 1:n-p
        read!(io, v)
        for j ∈ 1:length(v)
          x = v[j]
          setval!(G, 8*(j-1)+4, i, (x >> 3) & 0x01)
          setval!(G, 8*(j-1)+3, i, (x >> 2) & 0x01)
          setval!(G, 8*(j-1)+2, i, (x >> 1) & 0x01)
          setval!(G, 8*(j-1)+1, i, x & 0x01)
          setval!(G, 8*(j-1)+8, i, (x >> 7) & 0x01)
          setval!(G, 8*(j-1)+7, i, (x >> 6) & 0x01)
          setval!(G, 8*(j-1)+6, i, (x >> 5) & 0x01)
          setval!(G, 8*(j-1)+5, i, (x >> 4) & 0x01)
        end
      end
    catch ex
      throw(ErrorException("Bad generator: $(ex) - $(filename)"))
    end
    icols, G
  end
end

_run(cmd) = run(pipeline(cmd; stdout=devnull, stderr=devnull))

function _tool(name)
  candidates = (
    joinpath(@__DIR__, name),
    joinpath(@__DIR__, "..", "tools", "ldpc", name),
    name,
  )
  for candidate in candidates
    if candidate == name || isfile(candidate)
      return candidate
    end
  end
  error("could not find LDPC tool $(name)")
end

_tool_args(opts::AbstractString) = split(opts)
_tool_args(opts) = string.(collect(opts))

end # module
