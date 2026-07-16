# JUNA-lite: seed from the Partial-FFT combiner (so JUNA starts where Partial
# FFT+FEC ends), then re-fit the combiner toward BP posterior means as soft data
# anchors and keep the best re-decode.
function _juna_lite(m::Modulation, code::_Code, layout::_Layout, yparts, seed=nothing)
  _payload_from_metrics(m, code, _juna_lite_candidate(m, code, layout, yparts, seed).lpost_metric)
end

function _juna_lite_candidate(m::Modulation, code::_Code, layout::_Layout, yparts, seed=nothing)
  seed = seed === nothing ? _seed_candidate(m, code, layout, yparts) : seed
  seed.valid && return seed

  current = seed
  best = seed
  for _ in 1:_JUNA_ITERS
    candidate = _juna_step(m, code, layout, yparts, current)
    _juna_better(best, candidate) && (best = candidate)
    step_improves = _juna_better(current, candidate)
    current = candidate
    candidate.valid && break
    step_improves || break
  end

  best
end

function _juna_anchor_targets(m::Modulation,
                              layout::_Layout,
                              lpost_metric;
                              confidence_min::Real = _JUNA_CONFIDENCE_MIN,
                              max_data_anchors::Integer = _JUNA_MAX_DATA_ANCHORS)
  isfinite(confidence_min) && confidence_min >= 0 ||
    throw(ArgumentError("confidence_min must be finite and nonnegative"))
  max_data_anchors >= 0 ||
    throw(ArgumentError("max_data_anchors must be nonnegative"))

  anchors = _posterior_symbols(m, lpost_metric)
  confidence = _posterior_confidence(m, lpost_metric)
  n = min(length(anchors), length(layout.data_idx))
  selected = [i for i in 1:n if confidence[i] >= confidence_min]

  if length(selected) > max_data_anchors
    order = sortperm(confidence[selected]; rev=true)
    selected = selected[order[1:max_data_anchors]]
  end

  target_idx = vcat(layout.pilot_idx, layout.data_idx[selected])
  targets = vcat(layout.pilot_syms, ComplexF64.(anchors[selected]))
  target_weights = vcat(ones(Float64, length(layout.pilot_idx)), confidence[selected])
  (; target_idx, targets, target_weights, selected, confidence)
end

function _juna_step(m::Modulation, code::_Code, layout::_Layout, yparts, current)
  anchors = _juna_anchor_targets(m, layout, current.lpost_metric)
  equalized = _equalize_from_targets(
    m, yparts, layout, anchors.target_idx, anchors.targets;
    target_weights = anchors.target_weights,
  )
  _candidate_from_equalized(m, code, layout, equalized)
end
