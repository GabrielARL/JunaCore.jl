# JUNA modulation. Defaults reproduce the JOE paper red_1 controlled config
# n1024_cp256_sym1_p3_r0p25_dc4_sig1_ip2: QPSK, pilot every 3 active tones,
# LDPC rate 0.25, inner pilots every 2 message positions.
Base.@kwdef mutable struct Modulation <: Modulations.Modulation
  nc::UInt16 = 1024
  np::UInt16 = 256
  bw::Float64 = 1.0                    # occupied bandwidth as a fraction of fs
  dc0::Int16 = 0                       # occupied-band centre offset from fc, in kHz
  bpc::Int = 2                       # bits per data carrier: 1=BPSK, 2=QPSK
  pilot_ratio::Float64 = 1/3         # outer comb-pilot density (fraction of active tones); snapped to the nearest 1/k spacing
  inner_pilot_ratio::Float64 = 1/2   # inner-pilot density among message bits (0 = off); snapped to the nearest 1/k spacing
  sync::Bool = false                 # enable the selected sync/acquisition profile
  sync_profile::Symbol = :lfm        # :lfm legacy pre/postamble; :rpchan measured-replay preamble and acquisition
  compatibility_profile::Symbol = :juna # :juna native framing/layout; :rpchan paper-compatible framing/layout
  rpchan_preamble_seed::Int = 10_001 # ReplayCh uses frame_seed + 10_000; set this from an exported frame
  rpchan_guard_s::Float64 = 0.02
  rpchan_doppler_ppm::Float64 = 5_000.0
  rpchan_doppler_steps::Int = 81
  rpchan_sync_max_lag::Int = 400
  ldpc_k::Int = 340
  ldpc_n::Int = 1360
  ldpc_npc::Int = 3                  # dc: per-column check count passed to make-ldpc
  ldpc_seed::Int = 51_001            # frame/code seed used by the Rpchan-compatible systematic code
  partial_fft_parts::Int = 4
  partial_fft_nbands::Int = 16
  mode::Symbol = :lite               # receiver: canonical modes plus :frame_rls and the legacy :robust alias
  frame_receiver::Symbol = :stateful_lite # frame-wide FEC front end/refiner; preserves the original stateful receiver by default
  code::Any = nothing
  layout::Any = nothing
  bp_scratch::Any = nothing
end

const _MODE_STANDARD = :standard
const _MODE_PFFT = :pfft
const _MODE_LITE = :lite
const _MODE_FULL = :full
const _MODE_COUPLED = :coupled
const _MODE_FRAME_WIDE_LDPC = :frame_wide_ldpc
const _MODE_FRAME_RLS = :frame_rls
const _MODE_ROBUST = :robust
const _FRAME_RECEIVER_PROFILES =
  (_MODE_STANDARD, _MODE_PFFT, _MODE_LITE, _MODE_FULL, _MODE_COUPLED,
   :stateful_lite)
const _RECEIVER_PROFILES =
  (_MODE_STANDARD, _MODE_PFFT, _MODE_LITE, _MODE_FULL, _MODE_COUPLED,
   _MODE_FRAME_WIDE_LDPC)
const _PUBLIC_RECEIVER_MODES =
  (_MODE_STANDARD, _MODE_PFFT, _MODE_LITE, _MODE_FULL, _MODE_COUPLED,
   _MODE_FRAME_WIDE_LDPC, _MODE_FRAME_RLS)

receiver_profile(mode::Symbol) =
  mode === _MODE_ROBUST ? _MODE_FULL :
  mode === _MODE_FRAME_RLS ? _MODE_FRAME_WIDE_LDPC : mode
receiver_profile(m::Modulation) = receiver_profile(m.mode)

function Modulations.refinement_objective(m::Modulation)
  m.mode === _MODE_FRAME_RLS && return :frame_stateful_band_rls
  profile = receiver_profile(m)
  # The paper's benchmark baselines: :standard optimizes nothing (one-tap
  # interpolated equalization, declared :none), while :pfft's only objective
  # is the pilot-trained per-band ridge LS it solves in closed form
  # (eq:pfft-ls), so that is the capability it must prove executable.
  profile === _MODE_PFFT && return :pilot_band_ls
  profile === _MODE_LITE && return :posterior_anchor_ls
  profile === _MODE_FULL && return :reduced_wz
  profile === _MODE_COUPLED && return :coupled_cwz
  profile === _MODE_FRAME_WIDE_LDPC && return :frame_wide_ldpc
  :none
end

StandardModulation(; kwargs...) =
  Modulation(; (; kwargs..., mode = _MODE_STANDARD)...)
PartialFFTModulation(; kwargs...) =
  Modulation(; (; kwargs..., mode = _MODE_PFFT)...)
LiteModulation(; kwargs...) =
  Modulation(; (; kwargs..., mode = _MODE_LITE)...)
FullModulation(; kwargs...) =
  Modulation(; (; kwargs..., mode = _MODE_FULL)...)
CoupledModulation(; kwargs...) =
  Modulation(; (; kwargs..., mode = _MODE_COUPLED)...)
FrameWideLDPCModulation(; kwargs...) =
  Modulation(; (; kwargs..., mode = _MODE_FRAME_WIDE_LDPC)...)

function FrameRLSModulation(; kwargs...)
  identity = (
    mode=_MODE_FRAME_RLS,
    frame_receiver=:stateful_lite,
    sync=true,
    sync_profile=_SYNC_PROFILE_RPCHAN,
    compatibility_profile=_COMPATIBILITY_RPCHAN,
  )
  supplied = (; kwargs...)
  for (field, expected) in pairs(identity)
    haskey(supplied, field) || continue
    supplied[field] == expected || throw(ArgumentError(
      "FrameRLSModulation fixes $field=$expected"))
  end
  defaults = (
    nc=1024,
    np=16,
    bpc=2,
    pilot_ratio=1 / 5,
    inner_pilot_ratio=1 / 10,
    rpchan_preamble_seed=61_001,
    ldpc_k=817,
    ldpc_n=1634,
    ldpc_npc=10,
    ldpc_seed=51_001,
    partial_fft_parts=4,
    partial_fft_nbands=4,
  )
  Modulation(; merge(defaults, supplied, identity)...)
end

function _frame_receiver_profile(m::Modulation)
  receiver_profile(m) === _MODE_FRAME_WIDE_LDPC ||
    throw(ArgumentError("frame receiver profile only applies to frame-wide LDPC"))
  m.frame_receiver
end

# Fixed internal constants — folded out of the user-facing config (they are numerical
# defaults / solver internals nobody tunes per run). The _GRAD_* knobs only take effect
# when receiver_profile(m) === :full.
const _BP_ITERS = 20                           # belief-propagation iterations
const _JUNA_ITERS = 2                          # JUNA refinement passes
const _RIDGE = 1e-3                            # Tikhonov ridge on the RLS normal equations
const _LLR_CLIP = 20.0                         # channel-LLR clip magnitude
const _LLR_IP = 20.0                           # inner-pilot clamp magnitude
const _BP_ALPHA = 0.8                          # normalized min-sum scaling
const _JUNA_CONFIDENCE_MIN = 0.0               # min posterior confidence for a soft data anchor
const _JUNA_MAX_DATA_ANCHORS = typemax(Int)    # cap on soft data anchors used in the refit
const _GRAD_STEPS = 20
const _GRAD_LAMBDA_CODE = 0.08                 # parity-surrogate weight
const _GRAD_TRUST_MU = 50.0                    # trust region ‖z-z0‖
const _GRAD_GAMMA_Z = 1e-4                     # ridge on z
const _GRAD_ETA_W = 0.02                       # combiner anchor ‖W-W0‖
const _GRAD_TIE_WEIGHT = 1.0
const _GRAD_PILOT_WEIGHT = 2.0
const _GRAD_ALPHA_W = 0.006
const _GRAD_ALPHA_Z = 0.02
const _GRAD_CLIP_Z = 10.0
const _GRAD_CLIP_W = 25.0
const _GRAD_CLIP = 100.0
const _GRAD_BETA1 = 0.9
const _GRAD_BETA2 = 0.999
const _GRAD_EPS_ADAM = 1e-8
const _SYNC_LEN = 2048                         # LFM sync samples front+back when sync=true (best estimation in a len×bw×SNR sweep)
const _SYNC_BW = 0.9                           # chirp sweep as a fraction of the baseband band (sharp delay-Doppler peak, small guard)
const _SYNC_PROFILE_LFM = :lfm
const _SYNC_PROFILE_RPCHAN = :rpchan
const _SYNC_PROFILES = (_SYNC_PROFILE_LFM, _SYNC_PROFILE_RPCHAN)
const _COMPATIBILITY_JUNA = :juna
const _COMPATIBILITY_RPCHAN = :rpchan
const _COMPATIBILITY_PROFILES = (_COMPATIBILITY_JUNA, _COMPATIBILITY_RPCHAN)
const _RPCHAN_PREAMBLE_DURATION_S = 0.2
const _RPCHAN_PREAMBLE_GUARD_S = 0.02
const _RPCHAN_PREAMBLE_SEED = 10_001
const _RPCHAN_SYNC_MAX_LAG = 400
const _RPCHAN_DOPPLER_PPM = 5_000.0
const _RPCHAN_DOPPLER_STEPS = 81
const _RPCHAN_RESAMPLE_PHASES = 64
const _RPCHAN_RESAMPLE_HALF_SUPPORT = 8
const _LDPC_METHOD = "evencol"                 # make-ldpc construction
const _PARTIAL_FFT_NBANDS = 16                 # frequency bands for the bandwise RLS combiner
const _MAX_PARTIAL_FFT_PARTS = 16              # public complexity cap; band solves scale cubically in this count
const _BETA_FLOOR = 0.02                       # floor on the LLR-scale estimate from pilot residuals

struct _Code
  k::Int
  n::Int
  npc::Int
  method::String
  seed::Int
  icols::Vector{Int}
  gen::BitMatrix
  H::BitMatrix
  check_vars::Vector{Vector{Int}}              # check_vars[c] = variable indices in check c
  var_edges::Vector{Vector{Tuple{Int,Int}}}    # var_edges[v] = (check, local-index) edges
  invperm::Vector{Int}                         # undoes the systematic column permutation
end

struct _Layout
  signature::Tuple
  active::Vector{Int}
  pilot_idx::Vector{Int}
  data_idx::Vector{Int}
  pilot_syms::Vector{ComplexF64}
  bands::Vector{Vector{Int}}
  band_ids::Vector{Int}
  active_rank::Vector{Int}
end

struct _BPScratch
  signature::Tuple{Int,Int,Int,String,Int}
  lch::Vector{Float64}
  lpost::Vector{Float64}
  bits::Vector{Bool}
  q::Vector{Vector{Float64}}
  r::Vector{Vector{Float64}}
end

function Modulations.init(m::Modulation, fc, fs)
  _ = (fc, fs)
  frame_rls = m.mode === _MODE_FRAME_RLS
  m.nc = 1024
  m.np = frame_rls ? 16 : 256
  m.bw = 1.0
  m.dc0 = 0
  m.bpc = 2
  m.pilot_ratio = frame_rls ? 1/5 : 1/3
  m.inner_pilot_ratio = frame_rls ? 1/10 : 1/2
  m.ldpc_k = frame_rls ? 817 : 340
  m.ldpc_n = frame_rls ? 1634 : 1360
  m.ldpc_npc = frame_rls ? 10 : 3
  m.partial_fft_parts = 4
  m.partial_fft_nbands = frame_rls ? 4 : _PARTIAL_FFT_NBANDS
  # m.mode is left untouched so Modulation(mode=:robust) survives init()
  # JUNA-FrameRLS is an identity-bearing public preset; init restores it.
  if frame_rls
    m.frame_receiver = :stateful_lite
    m.sync = true
    m.sync_profile = _SYNC_PROFILE_RPCHAN
    m.compatibility_profile = _COMPATIBILITY_RPCHAN
  end
  # Other acquisition choices remain untouched so constructor-selected profiles survive init().
  m.code = nothing
  m.layout = nothing
  m.bp_scratch = nothing
  nothing
end

_bpc(m::Modulation) = Int(m.bpc)
_dc0_hz(m::Modulation) = 1_000.0 * Int(m.dc0)
_pm(b::Bool) = b ? -1.0 : 1.0    # bipolar map 1-2b: bit 0 -> +1, bit 1 -> -1

# Snap a pilot DENSITY ratio (fraction of positions that are pilots, e.g. 0.3) to the nearest
# achievable 1/k spacing ("every k-th position"), k >= kmin. ratio <= 0 means "off" (0).
function _ratio_spacing(ratio::Real, kmin::Int)
  ratio <= 0 && return 0
  inv = 1 / ratio
  klo = max(kmin, floor(Int, inv)); khi = klo + 1
  pick = abs(ratio - 1 / klo) <= abs(ratio - 1 / khi) ? klo : khi   # nearest unit fraction 1/k to the ratio
  max(kmin, pick)
end
_pilot_spacing(m::Modulation) = _ratio_spacing(m.pilot_ratio, 2)              # outer comb pilot every k-th active tone (k >= 2)
_inner_pilot_spacing(m::Modulation) = _ratio_spacing(m.inner_pilot_ratio, 1)  # inner pilot every k-th message bit (0 = off)

# Unknown payload bits per block = message bits minus inner pilots.
function Modulations.bitspersymbol(m::Modulation)
  k = Int(m.ldpc_k)
  ninner = _n_inner(m, k)
  k - ninner
end

_n_inner(m::Modulation, k::Integer) =
  (isp = _inner_pilot_spacing(m)) < 1 ? 0 : cld(Int(k), isp)

function Modulations.signallength(m::Modulation, nbits, fc, fs)
  isvalid(m, fc, fs) || throw(ArgumentError("invalid JUNA modulation settings"))
  nbits = _positive_nbits(nbits)
  _nblocks(m, nbits) * _blocklen(m) + _sync_overhead(m, fs)
end

function Base.isvalid(m::Modulation, fc, fs)
  fc isa Real && fs isa Real || return false
  isfinite(fc) && isfinite(fs) && fs > 0 || return false
  try
    isfinite(Float64(fc)) && isfinite(Float64(fs)) || return false
  catch
    return false
  end
  N = Int(m.nc)
  L = Int(m.np)
  count_ones(N) == 1 || return false
  N > 2 || return false
  0 <= L < N || return false
  _bpc(m) in (1, 2) || return false
  0 < m.bw <= 1 && isfinite(m.bw) || return false
  0 < m.ldpc_k < m.ldpc_n || return false
  0 < m.ldpc_npc <= m.ldpc_n - m.ldpc_k || return false
  0 < m.partial_fft_parts <= min(N, _MAX_PARTIAL_FFT_PARTS) || return false
  0 < m.partial_fft_nbands <= N || return false
  profile = receiver_profile(m)
  profile in _RECEIVER_PROFILES || return false
  profile in (_MODE_FULL, _MODE_COUPLED) && _bpc(m) != 2 && return false
  if profile === _MODE_FRAME_WIDE_LDPC
    m.frame_receiver in _FRAME_RECEIVER_PROFILES || return false
    m.frame_receiver in (_MODE_FULL, _MODE_COUPLED) && _bpc(m) != 2 &&
      return false
  end
  m.sync_profile in _SYNC_PROFILES || return false
  m.compatibility_profile in _COMPATIBILITY_PROFILES || return false
  m.compatibility_profile === _COMPATIBILITY_RPCHAN &&
    (m.bw != 1.0 || m.dc0 != 0) && return false
  m.ldpc_seed >= 0 || return false
  m.rpchan_preamble_seed >= 0 || return false
  isfinite(m.rpchan_guard_s) && m.rpchan_guard_s >= 0 || return false
  isfinite(m.rpchan_doppler_ppm) && m.rpchan_doppler_ppm >= 0 || return false
  m.rpchan_doppler_steps > 0 || return false
  m.rpchan_sync_max_lag >= 0 || return false
  0 < m.pilot_ratio <= 1 || return false      # pilot densities snap to a 1/k spacing (outer needs k >= 2)
  0 <= m.inner_pilot_ratio <= 1 || return false
  Modulations.bitspersymbol(m) > 0 || return false
  _pilot_spacing(m) > 1 || return false
  layout = _layout(m, fs)
  m.ldpc_n <= _bpc(m) * length(layout.data_idx) || return false
  _pilot_training_sufficient(m, layout) || return false
  if m.dc0 != 0                               # shifted occupied band must still fit within Nyquist
    nactive = clamp(floor(Int, (N - 1) * m.bw), 2, N - 1)
    (nactive ÷ 2) + abs(round(Int, _dc0_hz(m) / fs * N)) <= N ÷ 2 - 1 || return false
  end
  true
end

function _pilot_training_sufficient(m::Modulation, layout::_Layout)
  counts = zeros(Int, length(layout.bands))
  @inbounds for k in layout.pilot_idx
    band = layout.band_ids[k]
    band > 0 && (counts[band] += 1)
  end
  !isempty(counts) && minimum(counts) >= m.partial_fft_parts
end

# Transmit pipeline:
# 1. Pad and split payload bits into bitspersymbol-sized blocks.
# 2. Insert deterministic inner pilots into each k-bit LDPC message.
# 3. LDPC-encode each message from k information bits to n coded bits.
# 4. Map coded bits and outer pilots into the OFDM carrier layout.
# 5. IFFT, cyclic prefix, and normalization produce one complex baseband block.
# 6. Optional LFM sync wrapping produces [sync][blocks...][sync].
#    The Rpchan profile instead produces [preamble][guard][blocks...].
#
# How fc, fs, bw, and dc0 affect modulation: `fc` is the RF centre metadata for
# this complex-baseband waveform. `bw * fs` is the occupied RF bandwidth, and
# integer `dc0` shifts its centre away from `fc` in kHz. Thus fc=fs=24 kHz,
# bw=0.5, dc0=1 occupies approximately 19--31 kHz after upconversion. `fs` also
# sets the optional LFM waveform's time scale. With dc0=0 and sync=false, fc
# does not change the generated complex-baseband samples.
function Modulations.modulate(m::Modulation, bits, fc, fs)
  isvalid(m, fc, fs) || throw(ArgumentError("invalid JUNA modulation settings"))
  payload = Bool.(bits)
  isempty(payload) && throw(ArgumentError("JUNA modulation requires at least one payload bit"))
  receiver_profile(m) === _MODE_FRAME_WIDE_LDPC &&
    return _modulate_frame_wide_ldpc(m, payload, fs)
  code = _code(m)
  layout = _layout(m, fs)
  bps = Modulations.bitspersymbol(m)
  pad = mod(length(payload), bps)
  pad != 0 && append!(payload, falses(bps - pad))

  nblocks = div(length(payload), bps)
  out = Vector{ComplexF64}(undef, nblocks * _blocklen(m))
  for block in 1:nblocks
    blo = 1 + (block - 1) * bps
    bhi = block * bps
    message = _build_message(m, code, @view payload[blo:bhi])
    samples = _modulate_block(m, layout, _encode(code, message))
    copyto!(out, 1 + (block - 1) * _blocklen(m), samples, 1, _blocklen(m))
  end
  m.sync || return out
  if m.sync_profile === _SYNC_PROFILE_LFM
    sync = _sync_waveform(m, fs)
    return vcat(sync, out, sync)
  end
  preamble = _rpchan_preamble(m, fs)
  vcat(preamble, zeros(ComplexF64, _rpchan_guard_length(m, fs)), out)
end

function _prepare_demodulation(m::Modulation, nbits, x, fc, fs)
  isvalid(m, fc, fs) || throw(ArgumentError("invalid JUNA modulation settings"))
  nbits2 = _positive_nbits(nbits)
  waveform = _complex_waveform(x)
  _require_finite_waveform(waveform)
  code = _code(m)
  layout = _layout(m, fs)
  nblocks = _nblocks(m, nbits2)
  nbits2, waveform, code, layout, nblocks
end

function _require_block_samples(m::Modulation, waveform, nblocks::Integer)
  required = Int(nblocks) * _blocklen(m)
  length(waveform) >= required ||
    throw(ArgumentError("received $(length(waveform)) samples, need at least $required"))
  waveform
end

function Modulations.demodulate(m::Modulation, nbits, x, fc, fs)
  receiver_profile(m) === _MODE_FRAME_WIDE_LDPC &&
    return _demodulate_frame_wide_ldpc(m, nbits, x, fc, fs)
  nbits, waveform, code, layout, nblocks =
    _prepare_demodulation(m, nbits, x, fc, fs)
  cfo = 0.0
  if m.sync
    if m.sync_profile === _SYNC_PROFILE_LFM
      waveform, cfo = _coarse_doppler(m, waveform, fc, fs, nblocks)
    else
      acquired = _rpchan_acquire(m, waveform, fc, fs, nblocks)
      waveform, cfo = acquired.payload, acquired.cfo
    end
  end
  _require_block_samples(m, waveform, nblocks)

  metrics = Vector{Float64}(undef, Int(nbits))
  pos = 1
  for block in 1:nblocks
    lo = 1 + (block - 1) * _blocklen(m)
    hi = block * _blocklen(m)
    candidate = _demodulate_block_candidate(m, code, layout, @view waveform[lo:hi])
    pos = _write_payload_metrics!(metrics, pos, m, code, candidate.lpost_metric, Int(nbits))
  end

  metrics, cfo
end

function demodulate_methods(m::Modulation, nbits, x, fc, fs)
  receiver_profile(m) === _MODE_FRAME_WIDE_LDPC &&
    return _demodulate_frame_methods(m, nbits, x, fc, fs)
  nbits, waveform, code, layout, nblocks =
    _prepare_demodulation(m, nbits, x, fc, fs)
  _require_block_samples(m, waveform, nblocks)

  standard = Vector{Float64}(undef, Int(nbits))
  partial = Vector{Float64}(undef, Int(nbits))
  juna = Vector{Float64}(undef, Int(nbits))
  spos = ppos = jpos = 1
  for block in 1:nblocks
    lo = 1 + (block - 1) * _blocklen(m)
    hi = block * _blocklen(m)
    block = @view waveform[lo:hi]
    yparts = _branch_observations(m, block)
    standard_candidate = _standard_candidate(m, code, layout, yparts)
    seed = _seed_candidate(m, code, layout, yparts)
    juna_seed = _select_front_end_seed(standard_candidate, seed)
    juna_candidate = _juna_candidate(m, code, layout, yparts, juna_seed)
    spos = _write_payload_metrics!(standard, spos, m, code, standard_candidate.lpost_metric, Int(nbits))
    ppos = _write_payload_metrics!(partial, ppos, m, code, seed.lpost_metric, Int(nbits))
    jpos = _write_payload_metrics!(juna, jpos, m, code, juna_candidate.lpost_metric, Int(nbits))
  end

  (standard=standard, partial=partial, juna=juna)
end

_blocklen(m::Modulation) = Int(m.nc) + Int(m.np)
function _positive_nbits(nbits)
  nbits isa Integer && !(nbits isa Bool) ||
    throw(ArgumentError("nbits must be a positive integer"))
  0 < nbits <= typemax(Int) || throw(ArgumentError("nbits must be a positive integer"))
  Int(nbits)
end

_nblocks(m::Modulation, nbits::Integer) =
  receiver_profile(m) === _MODE_FRAME_WIDE_LDPC ?
    _frame_nblocks(m, nbits) :
    cld(_positive_nbits(nbits), Modulations.bitspersymbol(m))
_ndata_tones(m::Modulation, ncoded::Integer) = cld(Int(ncoded), _bpc(m))
_complex_waveform(x::AbstractVector{ComplexF64}) = x
_complex_waveform(x) = ComplexF64.(x)

function _require_finite_waveform(waveform)
  all(isfinite, waveform) ||
    throw(ArgumentError("received waveform must contain only finite samples"))
  waveform
end

# ---- coarse Doppler via a per-frame sync (LFM) pre/postamble ----------------
_synclen(m::Modulation) =
  m.sync && m.sync_profile === _SYNC_PROFILE_LFM ? _SYNC_LEN : 0

function _sync_overhead(m::Modulation, fs)
  m.sync || return 0
  m.sync_profile === _SYNC_PROFILE_LFM && return 2 * _SYNC_LEN
  length(_rpchan_preamble(m, fs)) + _rpchan_guard_length(m, fs)
end

# Deterministic baseband LFM (linear chirp) used as the per-frame sync pre/postamble.
function _sync_waveform(m::Modulation, fs)
  S = _synclen(m); S == 0 && return ComplexF64[]
  T = S / fs
  k = clamp(_SYNC_BW, 0.05, 1.0) * fs / T         # sweep rate (Hz/s): sweeps _SYNC_BW·fs over T
  ComplexF64[cispi(k * (n / fs - T / 2)^2) for n in 0:S-1]
end

# |matched filter| of rx against the known sync, over every lag.
function _matched_corr(rx::AbstractVector{<:Complex}, sync::AbstractVector{<:Complex})
  S = length(sync); M = length(rx); L = M - S + 1
  L <= 0 && return Float64[]
  sconj = conj.(sync)
  c = Vector{Float64}(undef, L)
  @inbounds for lag in 1:L
    acc = zero(ComplexF64)
    for i in 1:S
      acc += rx[lag + i - 1] * sconj[i]
    end
    c[lag] = abs(acc)
  end
  c
end

# Linear-interpolation resample of x to exactly `target` samples (coarse timing fix; the CP absorbs residual).
function _resample_to(x::AbstractVector{<:Complex}, target::Int)
  n = length(x)
  (target <= 0 || n == 0) && return ComplexF64[]
  n == target && return ComplexF64.(x)
  out = Vector{ComplexF64}(undef, target)
  @inbounds for i in 1:target
    pos = target == 1 ? 1.0 : 1 + (i - 1) * (n - 1) / (target - 1)
    lo = clamp(floor(Int, pos), 1, n); hi = min(lo + 1, n); frac = pos - lo
    out[i] = (1 - frac) * x[lo] + frac * x[hi]
  end
  out
end

# Coarse Doppler from two sync peaks. `fs` defines sample time and the sync
# waveform; the dimensionless observed/nominal spacing sets the resampling;
# `fc` converts that spacing error to the reported CFO in hertz.
function _coarse_doppler(m::Modulation, waveform::AbstractVector{<:Complex}, fc, fs, nblocks)
  S = _synclen(m); blocklen = _blocklen(m)
  nominal_blocks = nblocks * blocklen
  sync = _sync_waveform(m, fs)
  corr = _matched_corr(waveform, sync)
  length(corr) < 2 && return ComplexF64.(waveform), 0.0
  half = max(1, div(length(corr), 2))
  p1 = argmax(@view corr[1:half])                  # front sync start (1-based lag)
  p2 = half + argmax(@view corr[half + 1:end])     # back  sync start
  D0 = S + nominal_blocks                           # nominal start-to-start spacing
  D  = p2 - p1
  duration_scale = (D > 0 && D0 > 0) ? D / D0 : 1.0 # observed / nominal sync spacing
  bstart = p1 + round(Int, duration_scale * S)       # block region sits between the (dilated) syncs
  bstop  = p2 - 1
  (bstart < 1 || bstop > length(waveform) || bstop <= bstart) && return ComplexF64.(waveform), 0.0
  corrected = _resample_to(@view(waveform[bstart:bstop]), nominal_blocks)   # undo the dilation
  corrected, (duration_scale - 1) * fc
end

# ---- Rpchan-compatible preamble acquisition ---------------------------------

struct _RpchanResamplingFilterBank
  phases::Int
  offsets::UnitRange{Int}
  coefficients::Matrix{Float64}
end

const _RPCHAN_RESAMPLING_FILTER_BANKS =
  Dict{Tuple{Int,Int},_RpchanResamplingFilterBank}()

function _rpchan_preamble(m::Modulation, fs)
  m.sync && m.sync_profile === _SYNC_PROFILE_RPCHAN || return ComplexF64[]
  rate = Float64(fs)
  isfinite(rate) && rate > 0 || throw(ArgumentError("sample rate must be positive"))
  count = round(Int, _RPCHAN_PREAMBLE_DURATION_S * rate)
  count > 0 || throw(ArgumentError("sample rate produces an empty Rpchan preamble"))
  rng = MersenneTwister(m.rpchan_preamble_seed)
  alphabet = ComplexF64[
    (1 + 1im) / sqrt(2),
    (1 - 1im) / sqrt(2),
    (-1 + 1im) / sqrt(2),
    (-1 - 1im) / sqrt(2),
  ]
  ComplexF64[alphabet[rand(rng, eachindex(alphabet))] for _ in 1:count]
end

function _rpchan_guard_length(m::Modulation, fs)
  m.sync && m.sync_profile === _SYNC_PROFILE_RPCHAN || return 0
  rate = Float64(fs)
  isfinite(rate) && rate > 0 || throw(ArgumentError("sample rate must be positive"))
  round(Int, m.rpchan_guard_s * rate)
end

function _rpchan_doppler_grid(ppm::Real=_RPCHAN_DOPPLER_PPM,
                              steps::Integer=_RPCHAN_DOPPLER_STEPS)
  isfinite(ppm) && ppm >= 0 || throw(ArgumentError("Doppler range must be nonnegative"))
  steps > 0 || throw(ArgumentError("Doppler step count must be positive"))
  steps == 1 && return [1.0]
  collect(range(1 - Float64(ppm) * 1e-6, 1 + Float64(ppm) * 1e-6; length=Int(steps)))
end

_rpchan_doppler_grid(m::Modulation) =
  _rpchan_doppler_grid(m.rpchan_doppler_ppm, m.rpchan_doppler_steps)

@inline _rpchan_sinc(x::Float64) = abs(x) < 1e-14 ? 1.0 : sinpi(x) / (pi * x)
@inline function _rpchan_hann_sinc(x::Float64, half_support::Int)
  abs(x) >= half_support && return 0.0
  0.5 + 0.5 * cospi(abs(x) / half_support)
end

function _rpchan_filter_bank(phases::Int=_RPCHAN_RESAMPLE_PHASES,
                             half_support::Int=_RPCHAN_RESAMPLE_HALF_SUPPORT)
  phases > 0 || throw(ArgumentError("resampling phase count must be positive"))
  half_support > 0 || throw(ArgumentError("resampling support must be positive"))
  key = (phases, half_support)
  get!(_RPCHAN_RESAMPLING_FILTER_BANKS, key) do
    offsets = (-half_support + 1):half_support
    coefficients = Matrix{Float64}(undef, length(offsets), phases)
    for phase in 1:phases
      frac = (phase - 1) / phases
      total = 0.0
      for tap in eachindex(offsets)
        distance = frac - offsets[tap]
        weight = _rpchan_sinc(distance) * _rpchan_hann_sinc(distance, half_support)
        coefficients[tap, phase] = weight
        total += weight
      end
      abs(total) > 1e-15 && (@views coefficients[:, phase] ./= total)
    end
    _RpchanResamplingFilterBank(phases, offsets, coefficients)
  end
end

function _rpchan_resample(x::AbstractVector{<:Number}, scale::Real;
                           max_output_length=nothing,
                           filter_bank=_rpchan_filter_bank())
  isfinite(scale) && scale > 0 || throw(ArgumentError("time scale must be positive"))
  isempty(x) && throw(ArgumentError("cannot resample an empty sequence"))
  natural_length = floor(Int, (length(x) - 1) / Float64(scale)) + 1
  output_length = max_output_length === nothing ? natural_length :
                  min(natural_length, Int(max_output_length))
  output_length > 0 || throw(ArgumentError("resampled sequence would be empty"))
  scale == 1 && output_length == length(x) && return ComplexF64.(x)

  output = Vector{ComplexF64}(undef, output_length)
  offsets = filter_bank.offsets
  coefficients = filter_bank.coefficients
  phase_count = filter_bank.phases
  first_offset, last_offset = first(offsets), last(offsets)
  nx = length(x)
  for n in eachindex(output)
    position = 1 + (n - 1) * Float64(scale)
    center = floor(Int, position)
    fraction = position - center
    phase0 = round(Int, fraction * phase_count)
    phase = phase0 + 1
    if phase0 == phase_count
      center += 1
      phase = 1
    end

    accumulator = 0.0 + 0.0im
    weight_sum = 0.0
    if center + first_offset >= 1 && center + last_offset <= nx
      @inbounds for tap in axes(coefficients, 1)
        accumulator += x[center + offsets[tap]] * coefficients[tap, phase]
      end
      output[n] = accumulator
    else
      @inbounds for tap in axes(coefficients, 1)
        index = center + offsets[tap]
        if 1 <= index <= nx
          weight = coefficients[tap, phase]
          accumulator += x[index] * weight
          weight_sum += weight
        end
      end
      output[n] = abs(weight_sum) > 1e-15 ? accumulator / weight_sum :
                  ComplexF64(x[clamp(center, 1, nx)])
    end
  end
  output
end

function _rpchan_derotate!(waveform, correction_scale::Real, fc, fs)
  carrier_offset_hz = Float64(fc) * (1 - Float64(correction_scale))
  abs(carrier_offset_hz) < 1e-12 && return waveform
  rate = Float64(fs)
  @inbounds for n in eachindex(waveform)
    waveform[n] *= cispi(-2 * carrier_offset_hz * (n - 1) / rate)
  end
  waveform
end

function _rpchan_correlation_scores(received, reference; max_lag::Integer)
  nreference = length(reference)
  nreference > 0 || throw(ArgumentError("alignment reference must not be empty"))
  limit = min(Int(max_lag), length(received) - nreference)
  limit >= 0 || throw(ArgumentError("received sequence is shorter than its reference"))
  reference_power = sum(abs2, reference)
  reference_power > 0 || throw(ArgumentError("alignment reference must have nonzero power"))
  scores = Vector{Float64}(undef, limit + 1)
  window_power = sum(abs2, @view received[1:nreference])
  scores[1] = window_power
  @inbounds for lag in 1:limit
    window_power += abs2(received[lag + nreference]) - abs2(received[lag])
    scores[lag + 1] = window_power
  end
  Threads.@threads for lag in 0:limit
    accumulator = 0.0 + 0.0im
    @inbounds for index in 1:nreference
      accumulator += conj(reference[index]) * received[lag + index]
    end
    @inbounds begin
      power = scores[lag + 1]
      scores[lag + 1] = power == 0 ? 0.0 :
                        abs(accumulator) / sqrt(reference_power * power)
    end
  end
  scores
end

function _rpchan_align(received, reference; max_lag::Integer=_RPCHAN_SYNC_MAX_LAG)
  scores = _rpchan_correlation_scores(received, reference; max_lag=max_lag)
  lag = argmax(scores) - 1
  (lag=lag, score=scores[lag + 1], scores=scores)
end

function _rpchan_estimate_doppler(m::Modulation, received, preamble, fc, fs)
  max_output_length = length(preamble) + m.rpchan_sync_max_lag
  best = nothing
  for scale in _rpchan_doppler_grid(m)
    corrected = _rpchan_resample(
      received, scale; max_output_length=max_output_length,
    )
    _rpchan_derotate!(corrected, scale, fc, fs)
    aligned = _rpchan_align(
      corrected, preamble; max_lag=m.rpchan_sync_max_lag,
    )
    if best === nothing || aligned.score > best.score
      best = (
        scale=scale,
        ppm=(scale - 1) * 1e6,
        lag=aligned.lag,
        score=aligned.score,
      )
    end
  end
  best
end


function _rpchan_estimate_doppler(received, preamble, fc, fs)
  _rpchan_estimate_doppler(
    LiteModulation(sync=true, sync_profile=_SYNC_PROFILE_RPCHAN),
    received, preamble, fc, fs,
  )
end

function _rpchan_acquire(m::Modulation, waveform, fc, fs, nblocks::Integer)
  preamble = _rpchan_preamble(m, fs)
  estimate = _rpchan_estimate_doppler(m, waveform, preamble, fc, fs)
  corrected = _rpchan_resample(waveform, estimate.scale)
  _rpchan_derotate!(corrected, estimate.scale, fc, fs)
  aligned = _rpchan_align(corrected, preamble; max_lag=m.rpchan_sync_max_lag)
  payload_start = aligned.lag + length(preamble) + _rpchan_guard_length(m, fs) + 1
  payload_length = Int(nblocks) * _blocklen(m)
  payload_stop = payload_start + payload_length - 1
  payload_stop <= length(corrected) || throw(ArgumentError(
    "received sequence is too short after Rpchan preamble alignment"))
  (
    payload=copy(@view corrected[payload_start:payload_stop]),
    cfo=(estimate.scale - 1) * Float64(fc),
    scale=estimate.scale,
    ppm=estimate.ppm,
    lag=aligned.lag,
    score=aligned.score,
  )
end

function _write_metrics!(out::Vector{Float64}, pos::Int, payload, nbits::Int)
  @inbounds for bit in payload
    pos > nbits && break
    out[pos] = bit ? 1.0 : -1.0
    pos += 1
  end
  pos
end

function _write_payload_metrics!(out::Vector{Float64}, pos::Int, m::Modulation,
                                 code::_Code, metrics::AbstractVector{<:Real}, nbits::Int)
  mparity = code.n - code.k
  isp = _inner_pilot_spacing(m)
  @inbounds for p in 1:code.k
    isp >= 1 && (p - 1) % isp == 0 && continue
    pos > nbits && break
    out[pos] = metrics[code.invperm[mparity + p]] > 0 ? 1.0 : -1.0
    pos += 1
  end
  pos
end

function _layout(m::Modulation, fs)
  sig = (Int(m.nc), Float64(m.bw), _pilot_spacing(m), Int(m.partial_fft_nbands),
         Int(m.dc0), Float64(fs), m.compatibility_profile)
  m.layout isa _Layout && m.layout.signature == sig && return m.layout::_Layout

  N, bw, pilot_spacing, nbands, dc0_khz, fsr, compatibility = sig
  active = if compatibility === _COMPATIBILITY_RPCHAN
    half = N ÷ 2
    [mod(k, N) + 1 for k in vcat(collect((-half + 1):-1), collect(1:(half - 1)))]
  else
    nactive = clamp(floor(Int, (N - 1) * bw), 2, N - 1)
    npos = nactive ÷ 2
    nneg = nactive - npos
    bins = vcat(collect(2:1+npos), collect(N-nneg+1:N))        # nactive carriers centred on DC
    if dc0_khz != 0                                            # occupied-band centre offset from fc, in kHz
      shift = round(Int, (1_000.0 * dc0_khz) / fsr * N)        # baseband Hz → subcarrier bins
      shifted = Int[]
      for b in bins
        f = (b - 1) <= N ÷ 2 ? (b - 1) : (b - 1 - N)           # FFT bin → signed cycle-frequency
        fp = f + shift
        fp == 0 && continue                                    # keep DC nulled
        push!(shifted, mod(fp, N) + 1)
      end
      bins = sort!(unique(shifted))
    end
    bins
  end
  pilot_idx = compatibility === _COMPATIBILITY_RPCHAN ?
    active[1:pilot_spacing:end] :
    [k for k in active if (k - 2) % pilot_spacing == 0]
  pilot_set = Set(pilot_idx)
  data_idx = [k for k in active if !(k in pilot_set)]
  pilot_syms = compatibility === _COMPATIBILITY_RPCHAN ?
    fill(ComplexF64(1, 1) / sqrt(2), length(pilot_idx)) :
    ComplexF64[
      isodd((1103515245 * k + 12345) & 0x7fffffff) ? -1.0 + 0.0im : 1.0 + 0.0im
      for k in pilot_idx
    ]
  bands = Vector{Vector{Int}}(undef, nbands)
  for b in 1:nbands
    lo, hi = _part_bounds(length(active), nbands, b)
    bands[b] = collect(@view active[lo:hi])
  end
  band_ids = zeros(Int, N)
  active_rank = zeros(Int, N)
  for (rank, k) in enumerate(active)
    active_rank[k] = rank
  end
  for (band_id, band) in enumerate(bands)
    band_ids[band] .= band_id
  end

  m.layout = _Layout(sig, active, pilot_idx, data_idx, pilot_syms, bands,
                     band_ids, active_rank)
  m.layout::_Layout
end

function _code(m::Modulation)
  method = _code_method(m)
  seed = _code_seed(m, m.ldpc_k, m.ldpc_n, m.ldpc_npc)
  if m.code === nothing ||
      m.code.k != m.ldpc_k ||
      m.code.n != m.ldpc_n ||
      m.code.npc != m.ldpc_npc ||
      m.code.method != method ||
      m.code.seed != seed
    m.code = _create_code(m.ldpc_k, m.ldpc_n, m.ldpc_npc, method, seed)
    m.bp_scratch = nothing
  end
  m.code
end

# Build the LDPC code through the shared LDPC.jl builder. `ldpc_npc` (the per-column
# check count dc) is the configurable knob; the make-ldpc construction is fixed to the
# _LDPC_METHOD constant ("evencol") and threaded in here as the `method` argument.
_code_method(m::Modulation) =
  m.compatibility_profile === _COMPATIBILITY_RPCHAN ? "rpchan" : _LDPC_METHOD

_code_seed(m::Modulation, k::Integer, n::Integer, npc::Integer) =
  m.compatibility_profile === _COMPATIBILITY_RPCHAN ? m.ldpc_seed :
  _ldpc_seed(k, n, npc)

function _create_code(k::Int, n::Int, npc::Int, method::AbstractString,
                      seed::Int=_ldpc_seed(k, n, npc))
  0 < k < n || throw(ArgumentError("LDPC dimensions must satisfy 0 < k < n"))
  0 < npc <= n - k ||
    throw(ArgumentError("LDPC column degree must satisfy 0 < npc <= n-k"))
  method in ("rpchan", "frame_sparse") &&
    return _create_sparse_systematic_code(k, n, npc, seed, method)
  r = try
    LDPC.build(k, n; method=method, dc=npc, no4cycle=true, seed=seed)
  catch err
    err isa InterruptException && rethrow()
    detail = sprint(showerror, err)
    throw(ArgumentError("failed to construct LDPC ($k, $n, npc=$npc): $detail"))
  end
  check_vars, var_edges = _build_graph(r.H)
  _Code(k, n, npc, String(method), seed, r.icols, BitMatrix(r.gen), BitMatrix(r.H),
        check_vars, var_edges, invperm(r.icols))
end

function _create_sparse_systematic_code(k::Int, n::Int, dc::Int, seed::Int,
                                        method::AbstractString)
  m = n - k
  # Identity parity columns make encoding proportional to Tanner-graph edges.
  # This is the ReplayCh code shape and also keeps large frame-wide codes usable.
  column_degree = min(m, dc)
  rng = MersenneTwister(seed)
  generator = falses(m, k)
  H = falses(m, n)
  check_vars = [Int[] for _ in 1:m]
  chosen = Vector{Int}(undef, column_degree)
  @inbounds for column in 1:k
    used = 0
    while used < column_degree
      row = rand(rng, 1:m)
      any(i -> chosen[i] == row, 1:used) && continue
      used += 1
      chosen[used] = row
      generator[row, column] = true
      H[row, column] = true
      push!(check_vars[row], column)
    end
  end
  @inbounds for row in 1:m
    parity_variable = k + row
    H[row, parity_variable] = true
    push!(check_vars[row], parity_variable)
  end
  icols = vcat(collect(m+1:n), collect(1:m))
  var_edges = _build_var_edges(check_vars, n)
  _Code(k, n, dc, String(method), seed, icols, generator, H, check_vars,
        var_edges, invperm(icols))
end

_ldpc_seed(k, n, npc) = k * 1_000_000 + n * 1_000 + npc

function _build_graph(H)
  mrows, n = size(H)
  check_vars = [findall(@view H[c, :]) for c in 1:mrows]
  check_vars, _build_var_edges(check_vars, n)
end

function _build_var_edges(check_vars, n::Integer)
  var_edges = [Tuple{Int,Int}[] for _ in 1:n]
  for c in eachindex(check_vars)
    for (a, v) in enumerate(check_vars[c])
      push!(var_edges[v], (c, a))
    end
  end
  var_edges
end

# Systematic LDPC encoding: generator rows form the parity prefix while the
# original k message bits occupy the permuted systematic positions.
function _encode(code::_Code, bits::AbstractVector{Bool})
  length(bits) == code.k ||
    throw(ArgumentError("LDPC encoder expects $(code.k) bits, got $(length(bits))"))
  if code.method in ("rpchan", "frame_sparse")
    # For H=[G I], the transmitted order is [message; parity].
    codeword = falses(code.n)
    copyto!(codeword, 1, bits, 1, code.k)
    @inbounds for check in eachindex(code.check_vars)
      parity = false
      for variable in code.check_vars[check]
        variable <= code.k && (parity ⊻= bits[variable])
      end
      codeword[code.k + check] = parity
    end
    return codeword
  end
  nparity = code.n - code.k
  codeword = Vector{Bool}(undef, code.n)
  @inbounds for outpos in 1:code.n
    src = code.icols[outpos]
    if src <= nparity
      s = false
      for j in 1:code.k
        code.gen[src, j] && (s ⊻= bits[j])
      end
      codeword[outpos] = s
    else
      codeword[outpos] = bits[src - nparity]
    end
  end
  codeword
end

# ----- message <-> payload (inner pilots are known message bits) ---------------

_inner_bit(p::Integer) = isodd((1103515245 * p + 12345) & 0x7fffffff)

function _known_inner_bit(m::Modulation, message_position::Integer)
  if m.compatibility_profile === _COMPATIBILITY_RPCHAN
    spacing = _inner_pilot_spacing(m)
    pilot_number = div(Int(message_position) - 1, spacing) + 1
    return iseven(pilot_number)
  end
  _inner_bit(message_position)
end

# Expand one payload block into the k-bit LDPC message. Known inner-pilot bits
# occupy every isp-th message position; payload fills the positions between them.
function _build_message(m::Modulation, code::_Code, payload::AbstractVector{Bool})
  message = falses(code.k)
  max_payload = code.k - _n_inner(m, code.k)
  length(payload) <= max_payload ||
    throw(ArgumentError("block holds $(max_payload) payload bits, got $(length(payload))"))
  i = 1
  isp = _inner_pilot_spacing(m)
  for p in 1:code.k
    if isp >= 1 && (p - 1) % isp == 0
      message[p] = _inner_bit(p)
    elseif i <= length(payload)
      message[p] = payload[i]
      i += 1
    end
  end
  message
end

function _payload_from_metrics(m::Modulation, code::_Code, metrics::AbstractVector{<:Real})
  payload = Vector{Bool}(undef, code.k - _n_inner(m, code.k))
  mparity = code.n - code.k
  i = 1
  isp = _inner_pilot_spacing(m)
  @inbounds for p in 1:code.k
    isp >= 1 && (p - 1) % isp == 0 && continue
    payload[i] = metrics[code.invperm[mparity + p]] > 0
    i += 1
  end
  payload
end

# Inner-pilot clamps are written directly into the BP channel LLR buffer.
function _apply_inner_clamps!(m::Modulation, code::_Code, lch::Vector{Float64},
                              message_block_k::Int=code.k)
  isp = _inner_pilot_spacing(m)
  isp < 1 && return lch
  valid_block = 0 < message_block_k <= code.k && code.k % message_block_k == 0
  valid_block ||
    throw(ArgumentError("message block size must divide the LDPC message length"))
  mparity = code.n - code.k
  Lip = min(_LLR_CLIP, _LLR_IP)
  @inbounds for p in 1:code.k
    local_p = (p - 1) % message_block_k + 1
    (local_p - 1) % isp == 0 || continue
    lch[code.invperm[mparity + p]] = _known_inner_bit(m, local_p) ? -Lip : Lip
  end
  lch
end

# ----- modulation -------------------------------------------------------------

# BPSK consumes one coded bit per data tone. QPSK consumes an I/Q pair using
# bit 0 -> +1 and bit 1 -> -1, then divides by sqrt(2) for unit symbol power.
function _carrier_symbol(m::Modulation, codeword::AbstractVector{Bool}, tone::Int)
  if _bpc(m) == 1
    _bpsk_symbol(codeword[tone])
  else
    j = 2 * (tone - 1) + 1
    bI = codeword[j]
    bQ = j + 1 <= length(codeword) ? codeword[j + 1] : false
    ComplexF64(_pm(bI), _pm(bQ)) / sqrt(2)
  end
end

# Render one LDPC codeword as one CP-OFDM block: deterministic outer pilots and
# coded data fill frequency bins, IFFT creates N time samples, the final L samples
# become the cyclic prefix, and standard-deviation normalization fixes block scale.
function _modulate_block(m::Modulation, layout::_Layout, codeword::AbstractVector{Bool})
  carriers = zeros(ComplexF64, Int(m.nc))

  for (k, s) in zip(layout.pilot_idx, layout.pilot_syms)
    carriers[k] = s
  end

  ntones = _ndata_tones(m, length(codeword))
  for (i, k) in enumerate(layout.data_idx)
    carriers[k] = i <= ntones ? _carrier_symbol(m, codeword, i) : one(ComplexF64)
  end

  rpchan_compatible = m.compatibility_profile === _COMPATIBILITY_RPCHAN
  sym = ifft(carriers)
  rpchan_compatible && (sym .*= sqrt(Int(m.nc)))
  L = Int(m.np)
  N = Int(m.nc)
  block = Vector{ComplexF64}(undef, L + N)
  @inbounds for i in 1:L
    block[i] = sym[N - L + i]
  end
  @inbounds for i in 1:N
    block[L + i] = sym[i]
  end
  if !rpchan_compatible
    scale = std(block)
    @inbounds for i in eachindex(block)
      block[i] /= scale
    end
  end
  block
end

# ----- demodulation branches --------------------------------------------------

function _demodulate_block(m::Modulation, code::_Code, layout::_Layout, waveform)
  _payload_from_metrics(m, code, _demodulate_block_candidate(m, code, layout, waveform).lpost_metric)
end

function _demodulate_block_candidate(m::Modulation, code::_Code, layout::_Layout, waveform)
  yparts = _branch_observations(m, waveform)
  profile = receiver_profile(m)
  # Benchmark baselines stop at their own front end: :standard never pays for
  # the partial-FFT seed, and :pfft is the pure partial column of
  # demodulate_methods (no standard fallback, no refinement).
  profile === _MODE_STANDARD && return _standard_candidate(m, code, layout, yparts)
  profile === _MODE_PFFT && return _seed_candidate(m, code, layout, yparts)
  seed = _front_end_seed_candidate(m, code, layout, yparts)
  _juna_candidate(m, code, layout, yparts, seed)
end

function _demodulate_block_standard(m::Modulation, code::_Code, layout::_Layout, yparts)
  _payload_from_metrics(m, code, _standard_candidate(m, code, layout, yparts).lpost_metric)
end

function _demodulate_block_partial(m::Modulation, code::_Code, layout::_Layout, yparts)
  _payload_from_metrics(m, code, _seed_candidate(m, code, layout, yparts).lpost_metric)
end

function _standard_candidate(m::Modulation, code::_Code, layout::_Layout, yparts)
  equalized = _residual_pilot_equalize(m, layout, _sum_branches(yparts))
  _candidate_from_equalized(m, code, layout, equalized)
end

function _candidate_from_equalized(m::Modulation, code::_Code, layout::_Layout, equalized, metrics=nothing)
  if metrics === nothing
    metrics, pilot_mse = _channel_metrics_from_equalized(m, code.n, layout, equalized)
  else
    pilot_mse = _pilot_mse(layout, equalized)
  end
  _decode_candidate(m, code, layout, equalized, metrics, pilot_mse)
end

function _pilot_mse(layout::_Layout, equalized)
  pilot_sum = 0.0
  @inbounds for i in eachindex(layout.pilot_idx)
    pilot_sum += abs2(equalized[layout.pilot_idx[i]] - layout.pilot_syms[i])
  end
  pilot_sum / max(length(layout.pilot_idx), 1)
end

function _channel_metrics_from_equalized(m::Modulation, ncoded::Integer,
                                         layout::_Layout, equalized)
  n = Int(ncoded)
  ntones = _ndata_tones(m, n)
  pilot_mse = _pilot_mse(layout, equalized)
  beta = max(pilot_mse, _BETA_FLOOR)
  metrics = Vector{Float64}(undef, n)
  if _bpc(m) == 1
    for t in 1:n
      s = equalized[layout.data_idx[t]]
      metrics[t] = clamp((-2.0 * real(s)) / beta, -_LLR_CLIP, _LLR_CLIP)
    end
  else
    for t in 1:ntones
      s = equalized[layout.data_idx[t]]
      metrics[2t-1] = clamp((-2.0 * real(s)) / beta, -_LLR_CLIP, _LLR_CLIP)
      2t <= n && (metrics[2t] = clamp((-2.0 * imag(s)) / beta, -_LLR_CLIP, _LLR_CLIP))
    end
  end
  metrics, pilot_mse
end

# JUNA receiver dispatch: :lite (posterior-anchor RLS refit), :full
# (reduced-gradient Adam over W,z), or :coupled (joint C,W,z Adam). :robust is
# accepted as a legacy alias for :full.
function _demodulate_block_juna(m::Modulation, code::_Code, layout::_Layout, yparts, seed=nothing)
  _payload_from_metrics(m, code, _juna_candidate(m, code, layout, yparts, seed).lpost_metric)
end

function _juna_candidate(m::Modulation, code::_Code, layout::_Layout, yparts, seed=nothing)
  profile = receiver_profile(m)
  profile === _MODE_STANDARD && return _standard_candidate(m, code, layout, yparts)
  profile === _MODE_PFFT &&
    return seed === nothing ? _seed_candidate(m, code, layout, yparts) : seed
  profile === _MODE_FRAME_WIDE_LDPC &&
    return seed === nothing ? _seed_candidate(m, code, layout, yparts) : seed
  profile === _MODE_COUPLED && return _juna_wcz_candidate(m, code, layout, yparts, seed)
  profile === _MODE_FULL && return _juna_wz_candidate(m, code, layout, yparts, seed)
  _juna_lite_candidate(m, code, layout, yparts, seed)
end

function _seed_candidate(m::Modulation, code::_Code, layout::_Layout, yparts)
  equalized = _equalize_from_targets(m, yparts, layout, layout.pilot_idx, layout.pilot_syms)
  _candidate_from_equalized(m, code, layout, equalized)
end

function _select_front_end_seed(standard, partial)
  partial.valid && return partial
  standard.valid ? standard : partial
end

function _front_end_seed_candidate(m::Modulation, code::_Code, layout::_Layout, yparts)
  partial = _seed_candidate(m, code, layout, yparts)
  partial.valid && return partial
  _select_front_end_seed(_standard_candidate(m, code, layout, yparts), partial)
end

function _decode_candidate(m::Modulation, code::_Code, layout::_Layout, equalized, metrics, pilot_mse)
  bp = _bp_decode(m, code, metrics)
  tie_mse = _posterior_tie_mse(m, equalized, layout, bp.lpost_metric)
  syndrome_norm = bp.syndrome / max(size(code.H, 1), 1)
  mean_abs_lpost = mean(abs, bp.lpost_metric)
  score = pilot_mse + 0.25 * tie_mse + 0.05 * syndrome_norm - 1e-4 * mean_abs_lpost
  (
    lpost_metric=bp.lpost_metric,
    valid=bp.valid,
    syndrome=bp.syndrome,
    mean_abs_lpost=mean_abs_lpost,
    pilot_mse=pilot_mse,
    tie_mse=tie_mse,
    score=score,
  )
end

function _bp_check_normalized_min_sum!(out, incoming)
  length(out) == length(incoming) ||
    throw(DimensionMismatch("check-message buffers must have equal length"))
  L = length(incoming)
  L == 0 && return out
  if L == 1
    out[1] = _LLR_CLIP
    return out
  end

  signtot = 1.0
  min1 = Inf
  min2 = Inf
  argmin1 = 0
  @inbounds for a in 1:L
    value = incoming[a]
    signtot *= ifelse(value < 0.0, -1.0, 1.0)
    magnitude = abs(value)
    if magnitude < min1
      min2 = min1
      min1 = magnitude
      argmin1 = a
    elseif magnitude < min2
      min2 = magnitude
    end
  end
  @inbounds for a in 1:L
    sign_without_self = signtot * ifelse(incoming[a] < 0.0, -1.0, 1.0)
    magnitude = a == argmin1 ? min2 : min1
    out[a] = _BP_ALPHA * sign_without_self * magnitude
  end
  out
end

function _bp_check_sum_product!(out, incoming)
  length(out) == length(incoming) ||
    throw(DimensionMismatch("check-message buffers must have equal length"))
  L = length(incoming)
  L == 0 && return out
  if L == 1
    out[1] = _LLR_CLIP
    return out
  end
  limit = tanh(0.5 * _LLR_CLIP)
  @inbounds for a in 1:L
    product = 1.0
    for b in 1:L
      b == a && continue
      product *= tanh(0.5 * clamp(Float64(incoming[b]), -_LLR_CLIP, _LLR_CLIP))
    end
    out[a] = clamp(2atanh(clamp(product, -limit, limit)), -_LLR_CLIP, _LLR_CLIP)
  end
  out
end

# BP over the cached Tanner graph (array-based, no Dicts). The measured receiver
# uses normalized min-sum; the exact sum-product path is retained as a paper
# reference and executable cross-check.
function _bp_decode_impl(m::Modulation, code::_Code, metrics, check_update!;
                         message_block_k::Int=code.k)
  cv = code.check_vars
  ve = code.var_edges
  n = code.n
  bp = _bp_scratch(m, code)
  lch = bp.lch
  lpost = bp.lpost
  bits = bp.bits
  q = bp.q
  r = bp.r

  @inbounds for v in 1:n
    lch[v] = -Float64(metrics[v])               # channel LLR, positive = bit 0
    lpost[v] = lch[v]
    bits[v] = false
  end
  _apply_inner_clamps!(m, code, lch, message_block_k)

  @inbounds for c in eachindex(cv)
    qc = q[c]
    for a in eachindex(qc)
      qc[a] = lch[cv[c][a]]
    end
  end

  syndrome = typemax(Int)
  for _ in 1:_BP_ITERS
    for c in eachindex(cv)
      qc = q[c]
      rc = r[c]
      check_update!(rc, qc)
    end

    for v in 1:n
      total = lch[v]
      for (c, a) in ve[v]
        total += r[c][a]
      end
      lpost[v] = total
      bits[v] = total < 0.0
      for (c, a) in ve[v]
        q[c][a] = total - r[c][a]
      end
    end

    syndrome = _syndrome_weight(code, bits)
    syndrome == 0 && break
  end

  lpost_metric = Vector{Float64}(undef, n)
  @inbounds for v in 1:n
    lpost_metric[v] = -lpost[v]
  end
  (lpost_metric=lpost_metric, valid=syndrome == 0, syndrome=syndrome)
end

_bp_decode(m::Modulation, code::_Code, metrics) =
  _bp_decode_impl(m, code, metrics, _bp_check_normalized_min_sum!)

_bp_decode_sum_product(m::Modulation, code::_Code, metrics) =
  _bp_decode_impl(m, code, metrics, _bp_check_sum_product!)

function _bp_scratch(m::Modulation, code::_Code)::_BPScratch
  sig = (code.k, code.n, code.npc, code.method, code.seed)
  m.bp_scratch isa _BPScratch &&
    (m.bp_scratch::_BPScratch).signature == sig &&
    return m.bp_scratch::_BPScratch

  q = [Vector{Float64}(undef, length(vars)) for vars in code.check_vars]
  r = [Vector{Float64}(undef, length(vars)) for vars in code.check_vars]
  m.bp_scratch = _BPScratch(sig, zeros(Float64, code.n), zeros(Float64, code.n),
                            falses(code.n), q, r)
  m.bp_scratch::_BPScratch
end

function _syndrome_weight(code::_Code, bits::AbstractVector{Bool})
  cnt = 0
  for vars in code.check_vars
    s = false
    for v in vars
      s ⊻= bits[v]
    end
    s && (cnt += 1)
  end
  cnt
end

# ----- carrier / pilot geometry -----------------------------------------------

_bpsk_symbol(bit::Bool) = bit ? ComplexF64(-1.0, 0.0) : ComplexF64(1.0, 0.0)

function _branch_observations(m::Modulation, waveform)
  N = Int(m.nc)
  L = Int(m.np)
  yparts = Matrix{ComplexF64}(undef, m.partial_fft_parts, N)
  chunk = zeros(ComplexF64, N)

  for p in 1:m.partial_fft_parts
    lo, hi = _part_bounds(N, m.partial_fft_parts, p)
    fill!(chunk, 0.0 + 0.0im)
    @views chunk[lo:hi] .= waveform[L+lo:L+hi]
    fft!(chunk)
    @views yparts[p, :] .= chunk
  end

  yparts
end

function _equalize_from_targets(m::Modulation, yparts, layout::_Layout, target_idx, targets;
                                target_weights = nothing)
  _validate_target_weights(target_idx, target_weights)
  equalized = zeros(ComplexF64, Int(m.nc))
  P = m.partial_fft_parts
  A = Matrix{ComplexF64}(undef, P, P)
  b = Vector{ComplexF64}(undef, P)
  weights = Vector{ComplexF64}(undef, P)
  target_pos = zeros(Int, Int(m.nc))
  @inbounds for i in eachindex(target_idx)
    target_pos[target_idx[i]] = i
  end
  local_targets = Int[]
  sizehint!(local_targets, length(target_idx))

  for band in layout.bands
    empty!(local_targets)
    @inbounds for k in band
      pos = target_pos[k]
      pos == 0 || push!(local_targets, pos)
    end
    # A merely square local fit is numerically fragile for high branch counts:
    # use all pilots until the per-band system is at least 2x overdetermined.
    if length(local_targets) < 2 * m.partial_fft_parts
      resize!(local_targets, length(target_idx))
      @inbounds for i in eachindex(target_idx)
        local_targets[i] = i
      end
    end
    _fit_branch_weights!(
      weights, A, b, m, yparts, target_idx, targets, local_targets;
      target_weights = target_weights,
    )
    for k in band
      acc = 0.0 + 0.0im
      @inbounds for p in 1:m.partial_fft_parts
        acc += yparts[p, k] * weights[p]
      end
      equalized[k] = acc
    end
  end

  _residual_pilot_equalize(m, layout, equalized)
end

function _fit_branch_weights!(weights::Vector{ComplexF64}, A::Matrix{ComplexF64},
                              b::Vector{ComplexF64}, m::Modulation, yparts,
                              target_idx, targets, positions;
                              target_weights = nothing)
  _validate_target_weights(target_idx, target_weights)
  P = m.partial_fft_parts
  fill!(A, 0.0 + 0.0im)
  fill!(b, 0.0 + 0.0im)
  @inbounds for row in positions
    k = target_idx[row]
    target = ComplexF64(targets[row])
    row_weight = target_weights === nothing ? 1.0 : Float64(target_weights[row])
    for p in 1:P
      yp = yparts[p, k]
      cyp = conj(yp)
      b[p] += row_weight * cyp * target
      for q in 1:P
        A[p, q] += row_weight * cyp * yparts[q, k]
      end
    end
  end
  @inbounds for p in 1:P
    A[p, p] += _RIDGE
  end
  _solve_small!(weights, A, b)
end

function _validate_target_weights(target_idx, target_weights)
  target_weights === nothing && return nothing
  length(target_weights) == length(target_idx) ||
    throw(DimensionMismatch("target weights must match target indices"))
  all(weight -> isfinite(weight) && weight >= 0, target_weights) ||
    throw(ArgumentError("target weights must be finite and nonnegative"))
  nothing
end

function _solve_small!(x::Vector{ComplexF64}, A::Matrix{ComplexF64}, b::Vector{ComplexF64})
  n = length(x)
  copyto!(x, b)
  @inbounds for k in 1:n
    pivot = k
    pivot_abs = abs(A[k, k])
    for i in k+1:n
      cand = abs(A[i, k])
      if cand > pivot_abs
        pivot = i
        pivot_abs = cand
      end
    end
    if pivot != k
      for j in k:n
        A[k, j], A[pivot, j] = A[pivot, j], A[k, j]
      end
      x[k], x[pivot] = x[pivot], x[k]
    end
    akk = A[k, k]
    for i in k+1:n
      factor = A[i, k] / akk
      A[i, k] = 0.0 + 0.0im
      for j in k+1:n
        A[i, j] -= factor * A[k, j]
      end
      x[i] -= factor * x[k]
    end
  end
  @inbounds for i in n:-1:1
    acc = x[i]
    for j in i+1:n
      acc -= A[i, j] * x[j]
    end
    x[i] = acc / A[i, i]
  end
  x
end

function _residual_pilot_equalize(m::Modulation, layout::_Layout, carriers)
  equalized = carriers isa Vector{ComplexF64} ? carriers : ComplexF64.(carriers)
  response = Vector{ComplexF64}(undef, length(layout.pilot_idx))
  @inbounds for i in eachindex(layout.pilot_idx)
    response[i] = equalized[layout.pilot_idx[i]] / layout.pilot_syms[i]
  end
  pilot_positions = [layout.active_rank[k] for k in layout.pilot_idx]
  for k in layout.active
    h = _interp_response(pilot_positions, response, layout.active_rank[k])
    abs(h) > eps(Float64) && (equalized[k] /= h)
  end
  equalized
end

function _sum_branches(yparts)
  P, N = size(yparts)
  carriers = Vector{ComplexF64}(undef, N)
  @inbounds for k in 1:N
    acc = 0.0 + 0.0im
    for p in 1:P
      acc += yparts[p, k]
    end
    carriers[k] = acc
  end
  carriers
end

function _interp_response(pidx, response, k)
  pos = searchsortedlast(pidx, k)
  pos <= 0 && return response[1]
  pos >= length(pidx) && return response[end]
  t = (k - pidx[pos]) / (pidx[pos + 1] - pidx[pos])
  (1 - t) * response[pos] + t * response[pos + 1]
end

function _part_bounds(n::Int, nparts::Int, p::Int)
  base = div(n, nparts)
  extra = rem(n, nparts)
  lo = 1 + (p - 1) * base + min(p - 1, extra)
  hi = lo + base - 1 + (p <= extra ? 1 : 0)
  lo, hi
end

# ----- posterior soft information ---------------------------------------------

# Per-tone posterior-mean constellation points from posterior metrics.
function _posterior_symbols(m::Modulation, lpost_metric)
  if _bpc(m) == 1
    anchors = Vector{ComplexF64}(undef, length(lpost_metric))
    @inbounds for i in eachindex(lpost_metric)
      anchors[i] = ComplexF64(-tanh(0.5 * lpost_metric[i]), 0.0)
    end
    anchors
  else
    ntones = _ndata_tones(m, length(lpost_metric))
    anchors = Vector{ComplexF64}(undef, ntones)
    invsqrt2 = 1 / sqrt(2)
    @inbounds for t in 1:ntones
      base = 2t - 1
      xr = -tanh(0.5 * lpost_metric[base])
      xi = base + 1 <= length(lpost_metric) ? -tanh(0.5 * lpost_metric[base + 1]) : 0.0
      anchors[t] = ComplexF64(xr, xi) * invsqrt2
    end
    anchors
  end
end

# Per-tone confidence: BPSK |xi|, QPSK min(|xi_I|, |xi_Q|).
function _posterior_confidence(m::Modulation, lpost_metric)
  if _bpc(m) == 1
    confidence = Vector{Float64}(undef, length(lpost_metric))
    @inbounds for i in eachindex(lpost_metric)
      confidence[i] = abs(tanh(0.5 * lpost_metric[i]))
    end
    confidence
  else
    ntones = _ndata_tones(m, length(lpost_metric))
    confidence = Vector{Float64}(undef, ntones)
    @inbounds for t in 1:ntones
      base = 2t - 1
      xr = abs(tanh(0.5 * lpost_metric[base]))
      xi = base + 1 <= length(lpost_metric) ? abs(tanh(0.5 * lpost_metric[base + 1])) : xr
      confidence[t] = min(xr, xi)
    end
    confidence
  end
end

function _posterior_tie_mse(m::Modulation, equalized, layout::_Layout, lpost_metric)
  anchors = _posterior_symbols(m, lpost_metric)
  confidence = _posterior_confidence(m, lpost_metric)
  n = min(length(layout.data_idx), length(anchors), length(confidence))
  n == 0 && return Inf
  acc = 0.0
  weight_sum = 0.0
  @inbounds for i in 1:n
    weight = max(confidence[i], 1e-3)
    acc += weight * abs2(equalized[layout.data_idx[i]] - anchors[i])
    weight_sum += weight
  end
  acc / weight_sum
end

function _juna_better(base, candidate)
  candidate.valid != base.valid && return candidate.valid
  candidate.syndrome != base.syndrome && return candidate.syndrome < base.syndrome
  score_margin = 0.005 * max(abs(base.score), eps(Float64))
  candidate.score < base.score - score_margin && return true
  candidate.mean_abs_lpost > 1.01 * base.mean_abs_lpost + 1e-6
end
