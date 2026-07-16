# JunaCore.jl

The production Julia modules for the JUNA underwater acoustic OFDM modem.

```julia
using Pkg
Pkg.add(url="https://github.com/GabrielARL/JunaCore.jl")
using JunaCore
```

Receiver implementations live in `src/juna/`. Independent verification is in
[JunaCoreTests](https://github.com/GabrielARL/JunaCoreTests), and the browser UI
is in [JunaCoreExplorer](https://github.com/GabrielARL/JunaCoreExplorer).

Radford Neal's LDPC helpers are vendored under `tools/ldpc/`; see
`THIRD_PARTY_NOTICES.md`.
