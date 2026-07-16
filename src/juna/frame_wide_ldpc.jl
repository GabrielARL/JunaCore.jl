# Frame-wide FEC receiver. OFDM carrier mapping remains one block at a time,
# while a single LDPC codeword and BP graph span all blocks in the frame.

const _FRAME_RLS_FORGETTING = 0.98
const _FRAME_RLS_DELTA = 1e-2
const _FRAME_JUNA_ITERS = 4
const _FRAME_JUNA_CONFIDENCE_MIN = 0.0
const _FRAME_JUNA_MAX_DATA_ANCHORS = 1024
const _FRAME_BP_ITERS = 8
const _FRAME_LLR_CLIP = 50.0
const _FRAME_INNER_LLR = 50.0
const _FRAME_BIN_SIGMA2_SCALE = 4.0
const _FRAME_BIN_SIGMA2_MIN_BLOCKS = 8

function _frame_code(m::Modulation, nblocks::Integer)
  blocks = Int(nblocks)
  blocks > 0 || throw(ArgumentError("frame-wide LDPC needs at least one block"))
  k = blocks * Int(m.ldpc_k)
  n = blocks * Int(m.ldpc_n)
  method = _code_method(m)
  seed = _code_seed(m, k, n, m.ldpc_npc)
  if m.code === nothing ||
      m.code.k != k ||
      m.code.n != n ||
      m.code.npc != m.ldpc_npc ||
      m.code.method != method ||
      m.code.seed != seed
    m.code = _create_code(k, n, m.ldpc_npc, method, seed)
    m.bp_scratch = nothing
  end
  m.code::_Code
end

_frame_payload_capacity(m::Modulation, nblocks::Integer) = begin
  k = Int(nblocks) * Int(m.ldpc_k)
  k - _n_inner(m, k)
end

function _frame_nblocks(m::Modulation, nbits::Integer)
  required = _positive_nbits(nbits)
  blocks = max(1, cld(required, Modulations.bitspersymbol(m)))
  while blocks > 1 && _frame_payload_capacity(m, blocks - 1) >= required
    blocks -= 1
  end
  while _frame_payload_capacity(m, blocks) < required
    blocks += 1
  end
  blocks
end

_frame_inner_bit(m::Modulation, message_position::Integer) =
  _known_inner_bit(m, message_position)

function _build_frame_message(m::Modulation, code::_Code,
                              payload::AbstractVector{Bool}, nblocks::Integer)
  blocks = Int(nblocks)
  block_k = Int(m.ldpc_k)
  code.k == blocks * block_k ||
    throw(ArgumentError("frame LDPC message length does not match its OFDM block count"))
  capacity = _frame_payload_capacity(m, blocks)
  length(payload) <= capacity ||
    throw(ArgumentError("frame holds $capacity payload bits, got $(length(payload))"))

  message = falses(code.k)
  isp = _inner_pilot_spacing(m)
  payload_pos = 1
  @inbounds for p in 1:code.k
    if isp >= 1 && (p - 1) % isp == 0
      message[p] = _frame_inner_bit(m, p)
    elseif payload_pos <= length(payload)
      message[p] = payload[payload_pos]
      payload_pos += 1
    end
  end
  message
end

function _frame_payload_metrics(m::Modulation, code::_Code, metrics,
                                nblocks::Integer, nbits::Integer)
  blocks = Int(nblocks)
  block_k = Int(m.ldpc_k)
  code.k == blocks * block_k ||
    throw(ArgumentError("frame LDPC message length does not match its OFDM block count"))
  output = Vector{Float64}(undef, Int(nbits))
  mparity = code.n - code.k
  isp = _inner_pilot_spacing(m)
  output_pos = 1
  @inbounds for p in 1:code.k
    isp >= 1 && (p - 1) % isp == 0 && continue
    output_pos > nbits && break
    output[output_pos] = metrics[code.invperm[mparity + p]] > 0 ? 1.0 : -1.0
    output_pos += 1
  end
  output_pos == nbits + 1 ||
    throw(ArgumentError("frame LDPC did not expose the requested payload length"))
  output
end

function _modulate_frame_wide_ldpc(m::Modulation, payload::AbstractVector{Bool}, fs)
  nblocks = _frame_nblocks(m, length(payload))
  capacity = _frame_payload_capacity(m, nblocks)
  padded = copy(payload)
  append!(padded, falses(capacity - length(padded)))

  code = _frame_code(m, nblocks)
  layout = _layout(m, fs)
  message = _build_frame_message(m, code, padded, nblocks)
  codeword = _encode(code, message)
  block_n = Int(m.ldpc_n)
  out = Vector{ComplexF64}(undef, nblocks * _blocklen(m))
  for block in 1:nblocks
    coded_lo = 1 + (block - 1) * block_n
    coded_hi = block * block_n
    samples = _modulate_block(m, layout, @view codeword[coded_lo:coded_hi])
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

function _frame_rls_update!(weights, inverse_covariance, xraw, target;
                            forgetting::Real=_FRAME_RLS_FORGETTING,
                            scratch_x=similar(weights),
                            scratch_px=similar(weights),
                            scratch_gain=similar(weights),
                            scratch_row=similar(weights))
  lambda = Float64(forgetting)
  0 < lambda <= 1 || throw(ArgumentError(
    "frame RLS forgetting factor must satisfy 0 < lambda <= 1"))
  pcount = length(weights)
  length(xraw) == pcount || throw(DimensionMismatch(
    "frame RLS observation length must match its weights"))
  size(inverse_covariance) == (pcount, pcount) || throw(DimensionMismatch(
    "frame RLS inverse covariance must be square and match its weights"))

  @inbounds for i in 1:pcount
    scratch_x[i] = ComplexF64(xraw[i])
  end
  @inbounds for i in 1:pcount
    acc = 0.0 + 0.0im
    for j in 1:pcount
      acc += inverse_covariance[i, j] * scratch_x[j]
    end
    scratch_px[i] = acc
  end
  denominator = ComplexF64(lambda, 0.0)
  @inbounds for i in 1:pcount
    denominator += conj(scratch_x[i]) * scratch_px[i]
  end
  @inbounds for i in 1:pcount
    scratch_gain[i] = scratch_px[i] / denominator
  end

  predicted = 0.0 + 0.0im
  @inbounds for i in 1:pcount
    predicted += conj(weights[i]) * scratch_x[i]
  end
  residual = ComplexF64(target) - predicted
  @inbounds for i in 1:pcount
    weights[i] += scratch_gain[i] * conj(residual)
  end

  @inbounds for j in 1:pcount
    acc = 0.0 + 0.0im
    for i in 1:pcount
      acc += conj(scratch_x[i]) * inverse_covariance[i, j]
    end
    scratch_row[j] = acc
  end
  inv_lambda = inv(lambda)
  @inbounds for j in 1:pcount
    row = scratch_row[j]
    for i in 1:pcount
      inverse_covariance[i, j] =
        (inverse_covariance[i, j] - scratch_gain[i] * row) * inv_lambda
    end
  end
  weights
end

function _frame_nearest_valid_band(valid, band_id::Int)
  valid[band_id] && return band_id
  best = 0
  distance = typemax(Int)
  @inbounds for candidate in eachindex(valid)
    valid[candidate] || continue
    candidate_distance = abs(candidate - band_id)
    if candidate_distance < distance
      best = candidate
      distance = candidate_distance
    end
  end
  best > 0 || throw(ArgumentError("frame RLS has no trained pilot band"))
  best
end

function _frame_rls_band_ids(layout::_Layout, nbands::Int, nfft::Int)
  nbands > 0 || throw(ArgumentError("frame RLS needs at least one band"))
  nactive = length(layout.active)
  band_ids = zeros(Int, nfft)
  @inbounds for band_id in 1:min(nbands, nactive)
    lo = floor(Int, (band_id - 1) * nactive / nbands) + 1
    hi = floor(Int, band_id * nactive / nbands)
    for rank in lo:hi
      band_ids[layout.active[rank]] = band_id
    end
  end
  band_ids
end

function _frame_anchor_plan(m::Modulation, layout::_Layout, posterior_metrics,
                            block::Int)
  target_idx = copy(layout.pilot_idx)
  targets = copy(layout.pilot_syms)
  posterior_metrics === nothing && return target_idx, targets, 0

  block_n = Int(m.ldpc_n)
  lo = 1 + (block - 1) * block_n
  hi = block * block_n
  hi <= length(posterior_metrics) || throw(DimensionMismatch(
    "frame posterior does not cover OFDM block $block"))
  block_metrics = @view posterior_metrics[lo:hi]
  soft_symbols = _posterior_symbols(m, block_metrics)
  confidence = _posterior_confidence(m, block_metrics)
  count_data = min(length(layout.data_idx), length(soft_symbols), length(confidence))
  candidates = Tuple{Int,Float64,ComplexF64}[]
  sizehint!(candidates, count_data)
  @inbounds for data_position in 1:count_data
    c = confidence[data_position]
    c >= _FRAME_JUNA_CONFIDENCE_MIN || continue
    push!(candidates, (
      layout.data_idx[data_position], c, soft_symbols[data_position]))
  end
  if length(candidates) > _FRAME_JUNA_MAX_DATA_ANCHORS
    sort!(candidates; by=item -> item[2], rev=true)
    resize!(candidates, _FRAME_JUNA_MAX_DATA_ANCHORS)
  end
  for (carrier, _, target) in candidates
    push!(target_idx, carrier)
    push!(targets, target)
  end
  target_idx, targets, length(candidates)
end

function _frame_stateful_band_rls(m::Modulation, layout::_Layout, observations;
                                  posterior_metrics=nothing)
  ndims(observations) == 3 || throw(DimensionMismatch(
    "frame observations must have partial-FFT, carrier, and block axes"))
  pcount, nfft, nblocks = size(observations)
  pcount == m.partial_fft_parts || throw(DimensionMismatch(
    "frame observation branches do not match partial_fft_parts"))
  nfft == Int(m.nc) || throw(DimensionMismatch(
    "frame observation carriers do not match nc"))
  nblocks > 0 || throw(ArgumentError("frame RLS needs at least one OFDM block"))
  posterior_metrics === nothing ||
    length(posterior_metrics) == nblocks * Int(m.ldpc_n) ||
    throw(DimensionMismatch("frame posterior length does not match its OFDM blocks"))

  nbands = min(Int(m.partial_fft_nbands), length(layout.active))
  band_ids = _frame_rls_band_ids(layout, nbands, nfft)
  running_weights = zeros(ComplexF64, pcount, nbands)
  inverse_covariances = [
    Matrix{ComplexF64}(I, pcount, pcount) ./ _FRAME_RLS_DELTA
    for _ in 1:nbands
  ]
  valid = falses(nbands)
  combined = zeros(ComplexF64, nfft, nblocks)
  equalized = zeros(ComplexF64, nfft, nblocks)
  weight_history = Array{ComplexF64}(undef, pcount, nbands, nblocks)
  data_anchor_counts = zeros(Int, nblocks)
  scaled_x = Vector{ComplexF64}(undef, pcount)
  scratch_px = similar(scaled_x)
  scratch_gain = similar(scaled_x)
  scratch_row = similar(scaled_x)
  observation_scale = inv(sqrt(nfft))

  for block in 1:nblocks
    target_idx, targets, data_anchor_count =
      _frame_anchor_plan(m, layout, posterior_metrics, block)
    data_anchor_counts[block] = data_anchor_count
    @inbounds for (target_position, carrier) in pairs(target_idx)
      band_id = band_ids[carrier]
      band_id > 0 || continue
      for part in 1:pcount
        scaled_x[part] = observation_scale * observations[part, carrier, block]
      end
      _frame_rls_update!(
        @view(running_weights[:, band_id]),
        inverse_covariances[band_id],
        scaled_x,
        targets[target_position];
        scratch_x=scaled_x,
        scratch_px,
        scratch_gain,
        scratch_row,
      )
      valid[band_id] = true
    end

    applied_weights = copy(running_weights)
    for band_id in 1:nbands
      valid[band_id] && continue
      nearest = _frame_nearest_valid_band(valid, band_id)
      copyto!(@view(applied_weights[:, band_id]),
              @view(running_weights[:, nearest]))
    end
    weight_history[:, :, block] .= applied_weights
    @inbounds for carrier in layout.active
      band_id = band_ids[carrier]
      acc = 0.0 + 0.0im
      for part in 1:pcount
        acc += conj(applied_weights[part, band_id]) *
               (observation_scale * observations[part, carrier, block])
      end
      combined[carrier, block] = acc
    end
    equalized[:, block] .= _residual_pilot_equalize(
      m, layout, @view combined[:, block])
  end
  (
    equalized,
    combined,
    weights=weight_history,
    data_anchor_counts,
  )
end

function _frame_sigma2_floor(m::Modulation)
  m.compatibility_profile === _COMPATIBILITY_RPCHAN || return _BETA_FLOOR
  Int(m.nc) >= 2048 ? 0.5 : 1.0
end

function _frame_channel_metrics(m::Modulation, layout::_Layout, equalized)
  nblocks = size(equalized, 2)
  pilot_total = 0.0
  pilot_count = 0
  @inbounds for block in 1:nblocks
    for index in eachindex(layout.pilot_idx)
      carrier = layout.pilot_idx[index]
      pilot_total += abs2(equalized[carrier, block] - layout.pilot_syms[index])
      pilot_count += 1
    end
  end
  pilot_mse = pilot_total / max(pilot_count, 1)
  packet_sigma2 = max(pilot_mse, _frame_sigma2_floor(m))
  sigma2_by_carrier = fill(packet_sigma2, Int(m.nc))
  if _bpc(m) == 2 && nblocks >= _FRAME_BIN_SIGMA2_MIN_BLOCKS
    @inbounds for carrier in layout.data_idx
      residual = 0.0
      for block in 1:nblocks
        symbol = equalized[carrier, block]
        sliced = ComplexF64(real(symbol) >= 0 ? 1.0 : -1.0,
                            imag(symbol) >= 0 ? 1.0 : -1.0) / sqrt(2)
        residual += abs2(symbol - sliced)
      end
      sigma2_by_carrier[carrier] = max(
        _FRAME_BIN_SIGMA2_SCALE * residual / nblocks, packet_sigma2)
    end
  end

  block_n = Int(m.ldpc_n)
  metrics = Vector{Float64}(undef, nblocks * block_n)
  if _bpc(m) == 1
    @inbounds for block in 1:nblocks
      offset = (block - 1) * block_n
      for bit in 1:block_n
        carrier = layout.data_idx[bit]
        metrics[offset + bit] = clamp(
          -2real(equalized[carrier, block]) / sigma2_by_carrier[carrier],
          -_FRAME_LLR_CLIP, _FRAME_LLR_CLIP)
      end
    end
  else
    tones = cld(block_n, 2)
    @inbounds for block in 1:nblocks
      offset = (block - 1) * block_n
      for tone in 1:tones
        carrier = layout.data_idx[tone]
        symbol = equalized[carrier, block]
        denominator = sigma2_by_carrier[carrier]
        metrics[offset + 2tone - 1] = clamp(
          -2real(symbol) / denominator, -_FRAME_LLR_CLIP, _FRAME_LLR_CLIP)
        2tone <= block_n && (metrics[offset + 2tone] = clamp(
          -2imag(symbol) / denominator, -_FRAME_LLR_CLIP, _FRAME_LLR_CLIP))
      end
    end
  end
  metrics, pilot_mse
end

function _frame_apply_inner_clamps!(m::Modulation, code::_Code, lch)
  spacing = _inner_pilot_spacing(m)
  spacing < 1 && return lch
  parity = code.n - code.k
  clamp_abs = min(_FRAME_INNER_LLR, _FRAME_LLR_CLIP)
  @inbounds for position in 1:spacing:code.k
    variable = code.invperm[parity + position]
    lch[variable] = _frame_inner_bit(m, position) ? -clamp_abs : clamp_abs
  end
  lch
end

function _frame_bp_decode(m::Modulation, code::_Code, metrics)
  length(metrics) == code.n || throw(DimensionMismatch(
    "frame BP metrics must match the global LDPC code"))
  bp = _bp_scratch(m, code)
  lch, lpost, bits, q, r = bp.lch, bp.lpost, bp.bits, bp.q, bp.r
  @inbounds for variable in 1:code.n
    lch[variable] = -Float64(metrics[variable])
    lpost[variable] = lch[variable]
    bits[variable] = lch[variable] < 0
  end
  _frame_apply_inner_clamps!(m, code, lch)
  @inbounds for variable in 1:code.n
    lpost[variable] = lch[variable]
    bits[variable] = lch[variable] < 0
  end
  syndrome = _syndrome_weight(code, bits)
  if syndrome != 0
    @inbounds for check in eachindex(code.check_vars)
      for edge in eachindex(q[check])
        q[check][edge] = lch[code.check_vars[check][edge]]
      end
    end
    for _ in 1:_FRAME_BP_ITERS
      for check in eachindex(code.check_vars)
        _bp_check_normalized_min_sum!(r[check], q[check])
      end
      @inbounds for variable in 1:code.n
        total = lch[variable]
        for (check, edge) in code.var_edges[variable]
          total += r[check][edge]
        end
        lpost[variable] = total
        bits[variable] = total < 0
        for (check, edge) in code.var_edges[variable]
          q[check][edge] = total - r[check][edge]
        end
      end
      syndrome = _syndrome_weight(code, bits)
      syndrome == 0 && break
    end
  end
  lpost_metric = Vector{Float64}(undef, code.n)
  @inbounds for variable in 1:code.n
    lpost_metric[variable] = -lpost[variable]
  end
  (
    lpost_metric,
    bits=copy(bits),
    valid=syndrome == 0,
    syndrome,
  )
end

function _frame_tie_mse(m::Modulation, layout::_Layout, equalized, lpost_metric)
  nblocks = size(equalized, 2)
  block_n = Int(m.ldpc_n)
  total = 0.0
  weight_sum = 0.0
  for block in 1:nblocks
    lo = 1 + (block - 1) * block_n
    hi = block * block_n
    soft = _posterior_symbols(m, @view lpost_metric[lo:hi])
    confidence = _posterior_confidence(m, @view lpost_metric[lo:hi])
    count_data = min(length(layout.data_idx), length(soft), length(confidence))
    @inbounds for position in 1:count_data
      weight = max(confidence[position], 1e-3)
      total += weight * abs2(
        equalized[layout.data_idx[position], block] - soft[position])
      weight_sum += weight
    end
  end
  total / max(weight_sum, eps(Float64))
end

function _frame_candidate(m::Modulation, code::_Code, layout::_Layout, equalized)
  metrics, pilot_mse = _frame_channel_metrics(m, layout, equalized)
  bp = _frame_bp_decode(m, code, metrics)
  tie_mse = _frame_tie_mse(m, layout, equalized, bp.lpost_metric)
  mean_abs_lpost = mean(abs, bp.lpost_metric)
  syndrome_norm = bp.syndrome / max(size(code.H, 1), 1)
  (
    lpost_metric=bp.lpost_metric,
    channel_metrics=metrics,
    bits=bp.bits,
    valid=bp.valid,
    syndrome=bp.syndrome,
    mean_abs_lpost,
    pilot_mse,
    tie_mse,
    score=pilot_mse + 0.25tie_mse + 0.05syndrome_norm - 1e-4mean_abs_lpost,
  )
end

function _frame_juna_refine(m::Modulation, code::_Code, layout::_Layout,
                            observations)
  seed_fit = _frame_stateful_band_rls(m, layout, observations)
  seed = _frame_candidate(m, code, layout, seed_fit.equalized)
  seed.valid && return (
    seed,
    best=seed,
    seed_equalized=seed_fit.equalized,
    best_equalized=seed_fit.equalized,
    selected_iteration=0,
    data_anchor_counts=Int[],
  )

  current = seed
  best = seed
  best_equalized = seed_fit.equalized
  selected_iteration = 0
  all_anchor_counts = Int[]
  for iteration in 1:_FRAME_JUNA_ITERS
    fit = _frame_stateful_band_rls(
      m, layout, observations; posterior_metrics=current.lpost_metric)
    append!(all_anchor_counts, fit.data_anchor_counts)
    candidate = _frame_candidate(m, code, layout, fit.equalized)
    if _juna_better(best, candidate)
      best = candidate
      best_equalized = fit.equalized
      selected_iteration = iteration
    end
    step_improves = _juna_better(current, candidate)
    step_improves || break
    current = candidate
    candidate.valid && break
  end
  (
    seed,
    best,
    seed_equalized=seed_fit.equalized,
    best_equalized,
    selected_iteration,
    data_anchor_counts=all_anchor_counts,
  )
end

function _demodulate_frame_wide_ldpc(m::Modulation, nbits, x, fc, fs)
  isvalid(m, fc, fs) || throw(ArgumentError("invalid JUNA modulation settings"))
  nbits2 = _positive_nbits(nbits)
  waveform = _complex_waveform(x)
  _require_finite_waveform(waveform)
  nblocks = _frame_nblocks(m, nbits2)
  layout = _layout(m, fs)
  code = _frame_code(m, nblocks)

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

  observations = Array{ComplexF64}(
    undef, m.partial_fft_parts, Int(m.nc), nblocks)
  for block in 1:nblocks
    sample_lo = 1 + (block - 1) * _blocklen(m)
    sample_hi = block * _blocklen(m)
    observations[:, :, block] .= _branch_observations(
      m, @view waveform[sample_lo:sample_hi])
  end
  trace = _frame_juna_refine(m, code, layout, observations)
  _frame_payload_metrics(
    m, code, trace.best.lpost_metric, nblocks, nbits2), cfo
end
