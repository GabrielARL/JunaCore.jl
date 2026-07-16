# ----- reduced-gradient JUNA-Wz (paper §V.E) ---------------------------------
# Seed from the Partial-FFT combiner, then descend the reduced (W,z) objective
# with Adam, evaluating each step as a candidate and keeping the best decode.
function _juna_wz(m::Modulation, code::_Code, layout::_Layout, yparts, seed=nothing)
  _payload_from_metrics(m, code, _juna_wz_candidate(m, code, layout, yparts, seed).lpost_metric)
end

function _juna_wz_candidate(m::Modulation, code::_Code, layout::_Layout, yparts, seed=nothing)
  seed = seed === nothing ? _seed_candidate(m, code, layout, yparts) : seed

  best = seed
  current = _juna_wz_gradient_solve(m, code, layout, yparts, seed)
  _juna_better(best, current) && (best = current)
  best
end

struct _GradientScratch
  S::Vector{ComplexF64}
  xbit::Vector{Float64}
  gS::Vector{ComplexF64}
  gradx::Vector{Float64}
  prefix::Vector{Float64}
  clamped::Vector{Float64}
end

function _GradientScratch(m::Modulation, code::_Code)
  ntones = _ndata_tones(m, code.n)
  maxdeg = maximum((length(vars) for vars in code.check_vars); init=0)
  _GradientScratch(Vector{ComplexF64}(undef, ntones), Vector{Float64}(undef, code.n),
                   zeros(ComplexF64, ntones), zeros(Float64, code.n),
                   Vector{Float64}(undef, maxdeg), Vector{Float64}(undef, maxdeg))
end

function _juna_wz_gradient_solve(m::Modulation, code::_Code, layout::_Layout, yparts, seed)
  W = _initial_gradient_W(m, yparts, layout)
  W0 = copy(W)
  z = Vector{Float64}(undef, length(seed.lpost_metric))
  @inbounds for i in eachindex(z)
    z[i] = clamp(-Float64(seed.lpost_metric[i]), -_GRAD_CLIP_Z, _GRAD_CLIP_Z)
  end
  z0 = copy(z)
  confidence = _posterior_confidence(m, z0)
  parity_sets = code.check_vars

  gW = similar(W); gz = zeros(Float64, length(z))
  mW = zero(W); mz = zeros(Float64, length(z))
  vW = zeros(Float64, size(W)); vz = zeros(Float64, length(z))
  scratch = _GradientScratch(m, code)

  best_candidate = _gradient_candidate(m, code, layout, yparts, W, z)

  best_loss = _wz_loss_and_grad!(m, code, layout, gW, gz, W, z, W0, z0, yparts,
                                 confidence, parity_sets, scratch)
  best_W = copy(W); best_z = copy(z)

  for iter in 1:max(_GRAD_STEPS, 0)
    _adam_step_complex!(W, mW, vW, gW, iter, _GRAD_ALPHA_W, _GRAD_BETA1, _GRAD_BETA2,
                        _GRAD_EPS_ADAM, _GRAD_CLIP, _GRAD_CLIP_W)
    _adam_step_real!(z, mz, vz, gz, iter, _GRAD_ALPHA_Z, _GRAD_BETA1, _GRAD_BETA2,
                     _GRAD_EPS_ADAM, _GRAD_CLIP, _GRAD_CLIP_Z)
    loss = _wz_loss_and_grad!(m, code, layout, gW, gz, W, z, W0, z0, yparts,
                              confidence, parity_sets, scratch)
    if isfinite(loss) && loss < best_loss
      best_loss = loss; copyto!(best_W, W); copyto!(best_z, z)
    end
    candidate = _gradient_candidate(m, code, layout, yparts, W, z)
    _juna_better(best_candidate, candidate) && (best_candidate = candidate)
  end

  candidate = _gradient_candidate(m, code, layout, yparts, best_W, best_z)
  _juna_better(best_candidate, candidate) && (best_candidate = candidate)
  best_candidate
end

function _initial_gradient_W(m::Modulation, yparts, layout::_Layout)
  P = m.partial_fft_parts
  W = zeros(ComplexF64, P, length(layout.bands))
  A = Matrix{ComplexF64}(undef, P, P)
  b = Vector{ComplexF64}(undef, P)
  weights = Vector{ComplexF64}(undef, P)
  pilot_pos = zeros(Int, Int(m.nc))
  @inbounds for i in eachindex(layout.pilot_idx)
    pilot_pos[layout.pilot_idx[i]] = i
  end
  local_pilots = Int[]
  sizehint!(local_pilots, length(layout.pilot_idx))
  for (band_id, band) in enumerate(layout.bands)
    empty!(local_pilots)
    @inbounds for k in band
      pos = pilot_pos[k]
      pos == 0 || push!(local_pilots, pos)
    end
    if length(local_pilots) < m.partial_fft_parts
      resize!(local_pilots, length(layout.pilot_idx))
      @inbounds for i in eachindex(layout.pilot_idx)
        local_pilots[i] = i
      end
    end
    _fit_branch_weights!(weights, A, b, m, yparts, layout.pilot_idx, layout.pilot_syms, local_pilots)
    @inbounds for p in 1:P
      W[p, band_id] = conj(weights[p])
    end
  end
  W
end

function _gradient_candidate(m::Modulation, code::_Code, layout::_Layout, yparts, W, z)
  equalized = zeros(ComplexF64, Int(m.nc))
  for (band_id, band) in enumerate(layout.bands)
    for k in band
      acc = 0.0 + 0.0im
      for p in 1:m.partial_fft_parts
        acc += conj(W[p, band_id]) * yparts[p, k]
      end
      equalized[k] = acc
    end
  end
  metrics = Vector{Float64}(undef, length(z))
  @inbounds for i in eachindex(z)
    metrics[i] = clamp(-Float64(z[i]), -_LLR_CLIP, _LLR_CLIP)
  end
  _candidate_from_equalized(m, code, layout, equalized, metrics)
end

function _gradient_symbol_grid!(m::Modulation, S, xbit, z)
  _bpc(m) == 2 || throw(ArgumentError("JUNA-Wz implements the paper's QPSK branch; got bpc=$(m.bpc)"))
  invsqrt2 = 1 / sqrt(2)
  fill!(xbit, 0.0)
  for t in eachindex(S)
    base = 2t - 1
    xr = tanh(0.5 * z[base])
    xi = base + 1 <= length(z) ? tanh(0.5 * z[base + 1]) : 0.0
    xbit[base] = xr
    base + 1 <= length(z) && (xbit[base + 1] = xi)
    S[t] = ComplexF64(xr, xi) * invsqrt2
  end
  S
end

function _wz_loss_and_grad!(m::Modulation, code::_Code, layout::_Layout, gW, gz, W, z, W0, z0,
                            yparts, confidence, parity_sets, scratch)
  fill!(gW, 0.0 + 0.0im); fill!(gz, 0.0)
  fill!(scratch.gS, 0.0 + 0.0im); fill!(scratch.gradx, 0.0)
  _gradient_symbol_grid!(m, scratch.S, scratch.xbit, z)

  ntones = min(length(scratch.S), length(layout.data_idx), length(confidence))
  loss = 0.0

  tie_scale = _GRAD_TIE_WEIGHT / max(ntones, 1)
  for t in 1:ntones
    k = layout.data_idx[t]; band_id = layout.band_ids[k]
    xhat = 0.0 + 0.0im
    for p in 1:m.partial_fft_parts
      xhat += conj(W[p, band_id]) * yparts[p, k]
    end
    local_scale = tie_scale * confidence[t]
    local_scale <= 0.0 && continue
    r = xhat - scratch.S[t]
    loss += local_scale * abs2(r)
    for p in 1:m.partial_fft_parts
      gW[p, band_id] += local_scale * yparts[p, k] * conj(r)
    end
    scratch.gS[t] -= local_scale * r
  end

  pilot_scale = _GRAD_PILOT_WEIGHT / max(length(layout.pilot_idx), 1)
  for (i, k) in enumerate(layout.pilot_idx)
    band_id = layout.band_ids[k]
    xhat = 0.0 + 0.0im
    for p in 1:m.partial_fft_parts
      xhat += conj(W[p, band_id]) * yparts[p, k]
    end
    r = xhat - layout.pilot_syms[i]
    loss += pilot_scale * abs2(r)
    for p in 1:m.partial_fft_parts
      gW[p, band_id] += pilot_scale * yparts[p, k] * conj(r)
    end
  end

  invsqrt2 = 1 / sqrt(2)
  for t in 1:ntones
    base = 2t - 1
    xr = scratch.xbit[base]
    gz[base] += 2.0 * real(scratch.gS[t]) * (0.5 * (1.0 - xr * xr) * invsqrt2)
    if base + 1 <= length(z)
      xi = scratch.xbit[base + 1]
      gz[base + 1] += 2.0 * imag(scratch.gS[t]) * (0.5 * (1.0 - xi * xi) * invsqrt2)
    end
  end

  if _GRAD_LAMBDA_CODE > 0.0 && !isempty(parity_sets)
    lambda_scaled = _GRAD_LAMBDA_CODE / max(length(parity_sets), 1)
    loss += _parity_penalty_and_gradx!(scratch.gradx, scratch.xbit, parity_sets,
                                       lambda_scaled, scratch.prefix, scratch.clamped)
    for i in eachindex(z)
      dx = 0.5 * (1.0 - scratch.xbit[i] * scratch.xbit[i])
      gz[i] += scratch.gradx[i] * dx
    end
  end

  if _GRAD_ETA_W > 0.0
    scale = _GRAD_ETA_W / max(length(W), 1)
    for i in eachindex(W)
      d = W[i] - W0[i]; loss += scale * abs2(d); gW[i] += scale * d
    end
  end
  if _GRAD_GAMMA_Z > 0.0
    scale = _GRAD_GAMMA_Z / max(length(z), 1)
    for i in eachindex(z)
      loss += scale * z[i] * z[i]; gz[i] += 2.0 * scale * z[i]
    end
  end
  if _GRAD_TRUST_MU > 0.0
    scale = _GRAD_TRUST_MU / max(length(z), 1)
    for i in eachindex(z)
      d = z[i] - z0[i]; loss += 0.5 * scale * d * d; gz[i] += scale * d
    end
  end
  loss
end

function _parity_penalty_and_gradx!(gradx::Vector{Float64}, xbit::Vector{Float64},
                                    parity_sets::Vector{Vector{Int}}, lambda::Float64,
                                    prefix::Vector{Float64}, clamped::Vector{Float64})
  pen = 0.0
  @inbounds for inds in parity_sets
    d = length(inds); d == 0 && continue
    pf = 1.0
    for t in 1:d
      xi = clamp(xbit[inds[t]], -0.999, 0.999)
      clamped[t] = xi
      prefix[t] = pf
      pf *= xi
    end
    e = 1.0 - pf; pen += e * e
    scale = -2.0 * lambda * e; sf = 1.0
    for t in d:-1:1
      gradx[inds[t]] += scale * (prefix[t] * sf)
      sf *= clamped[t]
    end
  end
  lambda * pen
end

function _adam_step_complex!(x, m, v, g, iter::Int, alpha::Float64, beta1::Float64,
                             beta2::Float64, eps_adam::Float64, grad_clip::Float64, value_clip::Float64)
  b1c = 1.0 - beta1^iter; b2c = 1.0 - beta2^iter
  for i in eachindex(x)
    gi = g[i]
    gmag = abs(gi)
    gmag > grad_clip && (gi *= grad_clip / gmag)
    m[i] = beta1 * m[i] + (1.0 - beta1) * gi
    v[i] = beta2 * v[i] + (1.0 - beta2) * abs2(gi)
    x[i] -= alpha * (m[i] / b1c) / (sqrt(v[i] / b2c) + eps_adam)
    mag = abs(x[i]); mag > value_clip && (x[i] *= value_clip / mag)
  end
end

function _adam_step_real!(x::Vector{Float64}, m::Vector{Float64}, v::Vector{Float64},
                          g::Vector{Float64}, iter::Int, alpha::Float64, beta1::Float64,
                          beta2::Float64, eps_adam::Float64, grad_clip::Float64, value_clip::Float64)
  b1c = 1.0 - beta1^iter; b2c = 1.0 - beta2^iter
  for i in eachindex(x)
    gi = clamp(g[i], -grad_clip, grad_clip)
    m[i] = beta1 * m[i] + (1.0 - beta1) * gi
    v[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi
    x[i] = clamp(x[i] - alpha * (m[i] / b1c) / (sqrt(v[i] / b2c) + eps_adam), -value_clip, value_clip)
  end
end

