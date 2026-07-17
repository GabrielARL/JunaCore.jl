module JunaCore

include(joinpath(@__DIR__, "Modulations.jl"))
include(joinpath(@__DIR__, "LDPC.jl"))
include(joinpath(@__DIR__, "Juna.jl"))

module JunaLite
  using ..Juna
  export Modulation
  const Modulation = Juna.LiteModulation
end

module JunaFull
  using ..Juna
  export Modulation
  const Modulation = Juna.FullModulation
end

module JunaCoupled
  # Public facade: behavior is reached only through the Modulations interface.
  # Resolve the implementation once without binding Juna or solver internals here.
  export Modulation
  const Modulation = getfield(parentmodule(@__MODULE__), :Juna).CoupledModulation
end

module JunaStandard
  # Paper baseline: one-tap pilot-interpolated equalization + FEC, no refinement.
  export Modulation
  const Modulation = getfield(parentmodule(@__MODULE__), :Juna).StandardModulation
end

module JunaPartialFFT
  # Paper baseline: pilot-trained per-band partial-FFT combining + FEC (the
  # pure partial column of demodulate_methods, no standard fallback, no
  # refinement).
  export Modulation
  const Modulation = getfield(parentmodule(@__MODULE__), :Juna).PartialFFTModulation
end

module JunaFrameWideLDPC
  # Public facade: Partial-FFT carrier observations feed one LDPC/BP graph that
  # spans every OFDM block in the requested frame.
  export Modulation
  const Modulation = getfield(parentmodule(@__MODULE__), :Juna).FrameWideLDPCModulation
end

module JunaFrameRLS
  # Rpchan-compatible frame construction/acquisition with the paper receiver's
  # stateful band-RLS and frame-wide LDPC refinement path.
  export Modulation
  const Modulation = getfield(parentmodule(@__MODULE__), :Juna).FrameRLSModulation
end

end # module JunaCore
