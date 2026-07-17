# JunaCore.jl

The production Julia modules for the JUNA underwater acoustic OFDM modem.

```julia
using Pkg
Pkg.add(url="https://github.com/GabrielARL/JunaCore.jl")
using JunaCore
```

Public receivers expose the shared `Modulations.jl` interface through named
facades. The stateful frame receiver is available as JUNA-FrameRLS:

```julia
using JunaCore

m = JunaCore.JunaFrameRLS.Modulation()
bits = falses(JunaCore.Modulations.bitspersymbol(m))
x = JunaCore.Modulations.modulate(m, bits, 24_000.0, 9_600.0)
metrics, cfo = JunaCore.Modulations.demodulate(
    m, length(bits), x, 24_000.0, 9_600.0)
```

Its defaults pin the SG-1 Rpchan paper profile: N=1024, CP=16, outer/inner pilot
spacings 5/10, rate-1/2 Rpchan LDPC with column degree 10, Rpchan preamble and
Doppler acquisition, and the stateful frame-wide band-RLS/JUNA receiver. OFDM
and code geometry remain keyword-configurable for the other channel profiles;
the Rpchan framing, acquisition, and receiver identity cannot be overridden.

Receiver implementations live in `src/juna/`. Independent verification is in
[JunaCoreTests](https://github.com/GabrielARL/JunaCoreTests), and the browser UI
is in [JunaCoreExplorer](https://github.com/GabrielARL/JunaCoreExplorer).

## Frequency geometry

`bw` is the occupied bandwidth as a fraction of the full baseband sample rate
`fs`. `dc0` is an integer offset in kHz from the RF centre `fc`. The approximate
RF edges after upconversion are

```text
centre = fc + 1000 * dc0
width  = bw * fs
edges  = centre + (-width/2, +width/2)
```

For `fc=fs=24_000`, `bw=0.5`, and `dc0=0`, the occupied band is approximately
18--30 kHz. Setting `dc0=1` shifts the same band to approximately 19--31 kHz.
Actual edges are quantized to the nearest FFT bin and DC remains nulled.

Radford Neal's LDPC helpers are vendored under `tools/ldpc/`; see
`THIRD_PARTY_NOTICES.md`.
