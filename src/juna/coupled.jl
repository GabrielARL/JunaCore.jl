# JUNA-WCz objective foundations. The public solver uses the profiled band-tied
# parameterization; the same objective also supports the paper's ideal
# per-carrier C/W reference for equation and gradient validation.
Base.@kwdef struct _CoupledSolverSpec
  bits_per_symbol::Int = 2
  ofdm_symbols_per_solve::Int = 1
  ici_half_width::Int = 1
  response_parameterization::Symbol = :band_tied
  combiner_parameterization::Symbol = :band_tied
  inner_pilot_policy::Symbol = :hard_clamp
  filler_policy::Symbol = :fixed_known_symbol
  dc_policy::Symbol = :excluded
  inactive_neighbor_policy::Symbol = :zero
  normalization::Symbol = :per_term_count
  bp_projection::Bool = false
end

function _validate_coupled_solver_spec(spec::_CoupledSolverSpec)
  spec.bits_per_symbol == 2 ||
    throw(ArgumentError("coupled solver is QPSK-only"))
  spec.ofdm_symbols_per_solve == 1 ||
    throw(ArgumentError("coupled solver handles one OFDM symbol per solve"))
  spec.ici_half_width == 1 ||
    throw(ArgumentError("coupled solver requires ICI half-width K=1"))
  spec.response_parameterization in (:band_tied, :per_carrier) ||
    throw(ArgumentError("coupled response C must be band-tied or per-carrier"))
  spec.combiner_parameterization in (:band_tied, :per_carrier) ||
    throw(ArgumentError("coupled combiner W must be band-tied or per-carrier"))
  spec.inner_pilot_policy === :hard_clamp ||
    throw(ArgumentError("coupled inner pilots must be hard-clamped"))
  spec.filler_policy === :fixed_known_symbol ||
    throw(ArgumentError("coupled filler symbols must remain fixed and known"))
  spec.dc_policy === :excluded ||
    throw(ArgumentError("the DC carrier must be excluded from the coupled model"))
  spec.inactive_neighbor_policy === :zero ||
    throw(ArgumentError("inactive and out-of-band neighbors must contribute zero"))
  spec.normalization === :per_term_count ||
    throw(ArgumentError("coupled loss families must use per-term count normalization"))
  spec.bp_projection === false ||
    throw(ArgumentError("BP projection is disabled for the deterministic first solver"))
  spec
end

_coupled_ici_offsets(spec::_CoupledSolverSpec) =
  -spec.ici_half_width:spec.ici_half_width

const _COUPLED_SOLVER_SPEC =
  _validate_coupled_solver_spec(_CoupledSolverSpec())

# A miniature coupled problem keeps the physical carrier index as the common
# coordinate. Zero in neighbor_idx is the explicit sentinel for an inactive,
# DC, or out-of-range source carrier.
struct _CoupledProblem
  spec::_CoupledSolverSpec
  observations::Matrix{ComplexF64}
  active::Vector{Int}
  dc_index::Int
  pilot_idx::Vector{Int}
  pilot_syms::Vector{ComplexF64}
  data_idx::Vector{Int}
  filler_idx::Vector{Int}
  bands::Vector{Vector{Int}}
  band_ids::Vector{Int}
  neighbor_idx::Matrix{Int}
  symbol_slots::Vector{Int}
  fixed_symbols::Vector{ComplexF64}
  inner_pilot_mask::BitVector
  inner_pilot_bits::BitVector
  parity_sets::Vector{Vector{Int}}
  nbits::Int
end

_coupled_response_groups(problem::_CoupledProblem) =
  problem.spec.response_parameterization === :per_carrier ?
    length(problem.active) : length(problem.bands)

_coupled_combiner_groups(problem::_CoupledProblem) =
  problem.spec.combiner_parameterization === :per_carrier ?
    length(problem.active) : length(problem.bands)

function _coupled_active_slot(problem::_CoupledProblem, carrier::Integer)
  slot = searchsortedfirst(problem.active, Int(carrier))
  slot <= length(problem.active) && problem.active[slot] == carrier ||
    throw(ArgumentError("carrier $carrier is not active in the coupled problem"))
  slot
end

_coupled_response_group(problem::_CoupledProblem, carrier::Integer) =
  problem.spec.response_parameterization === :per_carrier ?
    _coupled_active_slot(problem, carrier) : problem.band_ids[carrier]

_coupled_combiner_group(problem::_CoupledProblem, carrier::Integer) =
  problem.spec.combiner_parameterization === :per_carrier ?
    _coupled_active_slot(problem, carrier) : problem.band_ids[carrier]

struct _CoupledState
  W::Matrix{ComplexF64}
  C::Array{ComplexF64,4}
  z::Vector{Float64}
end

struct _CoupledGradient
  W::Matrix{ComplexF64}
  C::Array{ComplexF64,4}
  z::Vector{Float64}
end

struct _CoupledTerms
  observation::Float64
  pilot::Float64
  tie::Float64
  response_regularization::Float64
  combiner_regularization::Float64
  smoothness::Float64
  parity::Float64
  total::Float64
end

struct _CoupledScratch
  symbols::Vector{ComplexF64}
  predicted_observations::Matrix{ComplexF64}
  combined_symbols::Vector{ComplexF64}
  relaxed_bits::Vector{Float64}
  symbol_gradient::Vector{ComplexF64}
  bit_gradient::Vector{Float64}
  parity_prefix::Vector{Float64}
  parity_clamped::Vector{Float64}
end

struct _CoupledWeights
  observation::Float64
  pilot::Float64
  tie::Float64
  response_regularization::Float64
  combiner_regularization::Float64
  smoothness::Float64
  parity::Float64
end

function _CoupledWeights(; observation::Real = 0.0,
                         pilot::Real = 0.0,
                         tie::Real = 0.0,
                         response_regularization::Real = 0.0,
                         combiner_regularization::Real = 0.0,
                         smoothness::Real = 0.0,
                         parity::Real = 0.0)
  values = Float64[
    observation,
    pilot,
    tie,
    response_regularization,
    combiner_regularization,
    smoothness,
    parity,
  ]
  all(isfinite, values) || throw(ArgumentError("coupled objective weights must be finite"))
  all(>=(0.0), values) ||
    throw(ArgumentError("coupled objective weights must be nonnegative"))
  _CoupledWeights(values...)
end

const _COUPLED_UNIT_WEIGHTS = _CoupledWeights(
  observation = 1,
  pilot = 1,
  tie = 1,
  response_regularization = 1,
  combiner_regularization = 1,
  smoothness = 1,
  parity = 1,
)

# Runtime policy: observation consistency must be able to overrule a misleading
# W-z tie, while pilots remain the strongest direct anchors. C/W regularization
# stays light and parity keeps the deployed W,z receiver's weight.
const _COUPLED_RUNTIME_WEIGHTS = _CoupledWeights(
  observation = 1.0,
  pilot = 2.0,
  tie = 0.1,
  response_regularization = 0.002,
  combiner_regularization = 0.02,
  smoothness = 0.02,
  parity = 0.08,
)

Base.@kwdef struct _CoupledOptimizerConfig
  steps::Int = 20
  alpha_W::Float64 = 0.006
  alpha_C::Float64 = 0.004
  alpha_z::Float64 = 0.02
  beta1::Float64 = 0.9
  beta2::Float64 = 0.999
  epsilon::Float64 = 1e-8
  gradient_clip::Float64 = 100.0
  complex_value_clip::Float64 = 25.0
  logit_clip::Float64 = 10.0
end

# The unconstrained logit coordinate is the most sensitive public update. Keep
# its step on the response-learning scale so eight fixed Adam steps improve the
# decoder candidate instead of overshooting a better early z checkpoint.
const _COUPLED_PUBLIC_CONFIG =
  _CoupledOptimizerConfig(steps = 8, alpha_z = 0.004)

Base.@kwdef struct _CoupledBCDConfig
  cycles::Int = 8
  step_W::Float64 = 1.0
  step_C::Float64 = 1.0
  step_z::Float64 = 1.0
  shrink::Float64 = 0.5
  min_step::Float64 = 1e-8
  complex_value_clip::Float64 = 25.0
  logit_clip::Float64 = 10.0
end

function _validate_positive_controls(label::AbstractString, controls)
  for (name, value) in controls
    isfinite(value) && value > 0.0 ||
      throw(ArgumentError("$label $name must be finite and positive"))
  end
  nothing
end

function _validate_coupled_optimizer_config(config::_CoupledOptimizerConfig)
  config.steps >= 0 || throw(ArgumentError("coupled optimizer steps must be nonnegative"))
  _validate_positive_controls("coupled optimizer", (
    (:alpha_W, config.alpha_W),
    (:alpha_C, config.alpha_C),
    (:alpha_z, config.alpha_z),
    (:epsilon, config.epsilon),
    (:gradient_clip, config.gradient_clip),
    (:complex_value_clip, config.complex_value_clip),
    (:logit_clip, config.logit_clip),
  ))
  0.0 <= config.beta1 < 1.0 ||
    throw(ArgumentError("coupled optimizer beta1 must lie in [0,1)"))
  0.0 <= config.beta2 < 1.0 ||
    throw(ArgumentError("coupled optimizer beta2 must lie in [0,1)"))
  config
end

function _validate_coupled_bcd_config(config::_CoupledBCDConfig)
  config.cycles >= 0 || throw(ArgumentError("coupled BCD cycles must be nonnegative"))
  _validate_positive_controls("coupled BCD", (
    (:step_W, config.step_W),
    (:step_C, config.step_C),
    (:step_z, config.step_z),
    (:min_step, config.min_step),
    (:complex_value_clip, config.complex_value_clip),
    (:logit_clip, config.logit_clip),
  ))
  isfinite(config.shrink) && 0.0 < config.shrink < 1.0 ||
    throw(ArgumentError("coupled BCD shrink must lie strictly between zero and one"))
  config
end

struct _CoupledSolveResult
  state::_CoupledState
  initial_loss::Float64
  best_loss::Float64
  loss_history::Vector{Float64}
  selected_iter::Int
end

function _coupled_indices(name::AbstractString, values, nc::Int)
  indices = Int[value for value in values]
  issorted(indices) || throw(ArgumentError("$name must be sorted"))
  allunique(indices) || throw(ArgumentError("$name must not contain duplicates"))
  all(k -> 1 <= k <= nc, indices) ||
    throw(ArgumentError("$name contains a carrier outside 1:$nc"))
  indices
end

function _CoupledProblem(observations::AbstractMatrix{<:Number};
                         active,
                         dc_index::Integer,
                         pilot_idx,
                         pilot_syms,
                         data_idx,
                         bands,
                         nbits::Integer,
                         inner_pilot_idx = Int[],
                         inner_pilot_bits = Bool[],
                         parity_sets = Vector{Vector{Int}}(),
                         filler_symbol::Number = 1.0 + 0.0im,
                         spec::_CoupledSolverSpec = _COUPLED_SOLVER_SPEC)
  _validate_coupled_solver_spec(spec)
  nparts, nc = size(observations)
  nparts > 0 || throw(ArgumentError("coupled observations need at least one branch"))
  nc > 0 || throw(ArgumentError("coupled observations need at least one carrier"))
  1 <= dc_index <= nc || throw(ArgumentError("dc_index must lie inside 1:$nc"))

  active2 = _coupled_indices("active", active, nc)
  isempty(active2) && throw(ArgumentError("the coupled active set must not be empty"))
  Int(dc_index) in active2 && throw(ArgumentError("the DC carrier must be excluded"))
  pilots = _coupled_indices("pilot_idx", pilot_idx, nc)
  data = _coupled_indices("data_idx", data_idx, nc)
  isempty(intersect(pilots, data)) ||
    throw(ArgumentError("pilot and data carriers must be disjoint"))
  sort(vcat(pilots, data)) == active2 ||
    throw(ArgumentError("pilot and data carriers must partition the active set"))

  pilot_symbols = ComplexF64[symbol for symbol in pilot_syms]
  length(pilot_symbols) == length(pilots) ||
    throw(DimensionMismatch("pilot_syms must match pilot_idx"))

  band_list = Vector{Vector{Int}}()
  for (band_id, band) in enumerate(bands)
    indices = _coupled_indices("bands[$band_id]", band, nc)
    isempty(indices) && throw(ArgumentError("coupled bands must not be empty"))
    push!(band_list, indices)
  end
  isempty(band_list) && throw(ArgumentError("the coupled model needs at least one band"))
  vcat(band_list...) == active2 ||
    throw(ArgumentError("bands must partition the ordered active set"))

  nbits2 = Int(nbits)
  nbits2 > 0 || throw(ArgumentError("the coupled latent vector must not be empty"))
  ntones = cld(nbits2, spec.bits_per_symbol)
  ntones <= length(data) ||
    throw(DimensionMismatch("data carriers cannot hold $nbits2 QPSK bits"))
  fillers = data[ntones + 1:end]

  band_ids = zeros(Int, nc)
  for (band_id, band) in enumerate(band_list), k in band
    band_ids[k] = band_id
  end

  active_mask = falses(nc)
  active_mask[active2] .= true
  offsets = collect(_coupled_ici_offsets(spec))
  neighbor_idx = zeros(Int, length(offsets), nc)
  for k in active2, (offset_pos, offset) in enumerate(offsets)
    q = k + offset
    if 1 <= q <= nc && q != dc_index && active_mask[q]
      neighbor_idx[offset_pos, k] = q
    end
  end

  symbol_slots = zeros(Int, nc)
  for slot in 1:ntones
    symbol_slots[data[slot]] = slot
  end
  fixed_symbols = zeros(ComplexF64, nc)
  fixed_symbols[pilots] .= pilot_symbols
  fixed_symbols[fillers] .= ComplexF64(filler_symbol)

  inner_indices = _coupled_indices("inner_pilot_idx", inner_pilot_idx, nbits2)
  known_bits = Bool[bit for bit in inner_pilot_bits]
  length(known_bits) == length(inner_indices) ||
    throw(DimensionMismatch("inner_pilot_bits must match inner_pilot_idx"))
  inner_mask = falses(nbits2)
  inner_bits = falses(nbits2)
  inner_mask[inner_indices] .= true
  inner_bits[inner_indices] .= known_bits

  checks = Vector{Vector{Int}}()
  for (check_id, variables) in enumerate(parity_sets)
    check = _coupled_indices("parity_sets[$check_id]", variables, nbits2)
    isempty(check) && throw(ArgumentError("parity sets must not be empty"))
    push!(checks, check)
  end

  _CoupledProblem(
    spec,
    Matrix{ComplexF64}(observations),
    active2,
    Int(dc_index),
    pilots,
    pilot_symbols,
    data,
    fillers,
    band_list,
    band_ids,
    neighbor_idx,
    symbol_slots,
    fixed_symbols,
    inner_mask,
    inner_bits,
    checks,
    nbits2,
  )
end

function _CoupledState(problem::_CoupledProblem;
                       W::AbstractMatrix = zeros(ComplexF64,
                                                 size(problem.observations, 1),
                                                 _coupled_combiner_groups(problem)),
                       C::AbstractArray{<:Number,4} = zeros(
                         ComplexF64,
                         size(problem.observations, 1),
                         length(_coupled_ici_offsets(problem.spec)),
                         _coupled_response_groups(problem),
                         problem.spec.ofdm_symbols_per_solve,
                       ),
                       z::AbstractVector = zeros(Float64, problem.nbits))
  expected_W = (size(problem.observations, 1), _coupled_combiner_groups(problem))
  expected_C = (
    size(problem.observations, 1),
    length(_coupled_ici_offsets(problem.spec)),
    _coupled_response_groups(problem),
    problem.spec.ofdm_symbols_per_solve,
  )
  size(W) == expected_W ||
    throw(DimensionMismatch("W must have shape $expected_W"))
  size(C) == expected_C ||
    throw(DimensionMismatch("C must have shape $expected_C"))
  length(z) == problem.nbits ||
    throw(DimensionMismatch("z must have length $(problem.nbits)"))

  C2 = Array{ComplexF64,4}(undef, size(C)...)
  copyto!(C2, C)
  _CoupledState(Matrix{ComplexF64}(W), C2, Float64[value for value in z])
end

function _CoupledGradient(problem::_CoupledProblem)
  nparts = size(problem.observations, 1)
  _CoupledGradient(
    zeros(ComplexF64, nparts, _coupled_combiner_groups(problem)),
    zeros(ComplexF64,
          nparts,
          length(_coupled_ici_offsets(problem.spec)),
          _coupled_response_groups(problem),
          problem.spec.ofdm_symbols_per_solve),
    zeros(Float64, problem.nbits),
  )
end

# Translate the deployed one-block receiver geometry into the constrained
# coupled coordinate system. Inner pilots are message positions, so map them
# through the LDPC systematic permutation before they become z clamps.
function _coupled_problem_from_receiver(m::Modulation,
                                        code::_Code,
                                        layout::_Layout,
                                        yparts::AbstractMatrix{<:Number})
  _bpc(m) == 2 || throw(ArgumentError("JUNA-WCz initialization is QPSK-only"))
  expected = (Int(m.partial_fft_parts), Int(m.nc))
  size(yparts) == expected ||
    throw(DimensionMismatch("coupled branch observations must have shape $expected"))
  code.n <= 2 * length(layout.data_idx) ||
    throw(DimensionMismatch("QPSK data carriers cannot hold $(code.n) coded bits"))

  inner_pairs = Tuple{Int,Bool}[]
  spacing = _inner_pilot_spacing(m)
  nparity = code.n - code.k
  if spacing >= 1
    @inbounds for message_pos in 1:code.k
      (message_pos - 1) % spacing == 0 || continue
      push!(inner_pairs, (
        code.invperm[nparity + message_pos],
        _inner_bit(message_pos),
      ))
    end
    sort!(inner_pairs; by = first)
  end

  _CoupledProblem(
    yparts;
    active = layout.active,
    dc_index = 1,
    pilot_idx = layout.pilot_idx,
    pilot_syms = layout.pilot_syms,
    data_idx = layout.data_idx,
    bands = layout.bands,
    nbits = code.n,
    inner_pilot_idx = first.(inner_pairs),
    inner_pilot_bits = last.(inner_pairs),
    parity_sets = code.check_vars,
  )
end

function _profile_initial_coupled_C!(C,
                                     problem::_CoupledProblem,
                                     symbols::AbstractVector{<:Complex};
                                     ridge::Float64 = _RIDGE)
  noffsets = size(problem.neighbor_idx, 1)
  gram = Matrix{ComplexF64}(undef, noffsets, noffsets)
  rhs = Vector{ComplexF64}(undef, noffsets)
  design = Vector{ComplexF64}(undef, noffsets)

  target_groups = problem.spec.response_parameterization === :per_carrier ?
                  [[k] for k in problem.active] : problem.bands
  @inbounds for (group_id, targets) in enumerate(target_groups)
    for branch in axes(problem.observations, 1)
      fill!(gram, 0.0 + 0.0im)
      fill!(rhs, 0.0 + 0.0im)
      for k in targets
        for offset_pos in 1:noffsets
          q = problem.neighbor_idx[offset_pos, k]
          design[offset_pos] = q == 0 ? 0.0 + 0.0im : symbols[q]
        end
        observation = problem.observations[branch, k]
        for row in 1:noffsets
          conjugate = conj(design[row])
          rhs[row] += conjugate * observation
          for column in 1:noffsets
            gram[row, column] += conjugate * design[column]
          end
        end
      end
      for offset_pos in 1:noffsets
        gram[offset_pos, offset_pos] += ridge
      end
      C[branch, :, group_id, 1] .= gram \ rhs
    end
  end
  C
end

# Deterministic feasible start: pilot-trained W, BP-seeded bit logits, and a
# bandwise ridge profile of C conditional on those relaxed symbols.
function _initial_coupled_state(m::Modulation,
                                code::_Code,
                                layout::_Layout,
                                problem::_CoupledProblem,
                                seed)
  hasproperty(seed, :lpost_metric) ||
    throw(ArgumentError("coupled initialization requires seed posterior metrics"))
  metrics = seed.lpost_metric
  length(metrics) == problem.nbits ||
    throw(DimensionMismatch("seed metrics must have length $(problem.nbits)"))
  all(isfinite, problem.observations) ||
    throw(ArgumentError("coupled initialization requires finite observations"))
  all(isfinite, metrics) ||
    throw(ArgumentError("coupled initialization requires finite posterior metrics"))
  problem.nbits == code.n ||
    throw(DimensionMismatch("coupled problem bits must match the LDPC code length"))
  problem.active == layout.active ||
    throw(ArgumentError("coupled problem active carriers do not match the layout"))

  band_W = _initial_gradient_W(m, problem.observations, layout)
  W = if problem.spec.combiner_parameterization === :per_carrier
    expanded = zeros(ComplexF64, size(band_W, 1), length(problem.active))
    @inbounds for (slot, k) in enumerate(problem.active)
      expanded[:, slot] .= band_W[:, problem.band_ids[k]]
    end
    expanded
  else
    band_W
  end
  z = Vector{Float64}(undef, problem.nbits)
  @inbounds for bit_idx in eachindex(z)
    z[bit_idx] = clamp(-Float64(metrics[bit_idx]), -_GRAD_CLIP_Z, _GRAD_CLIP_Z)
    if problem.inner_pilot_mask[bit_idx]
      z[bit_idx] = problem.inner_pilot_bits[bit_idx] ? -_GRAD_CLIP_Z : _GRAD_CLIP_Z
    end
  end

  state = _CoupledState(problem; W = W, z = z)
  scratch = _CoupledScratch(problem)
  _coupled_symbols!(problem, state, scratch)
  _profile_initial_coupled_C!(state.C, problem, scratch.symbols)

  all(x -> isfinite(real(x)) && isfinite(imag(x)), state.W) ||
    throw(ArgumentError("coupled W initialization produced a non-finite value"))
  all(x -> isfinite(real(x)) && isfinite(imag(x)), state.C) ||
    throw(ArgumentError("coupled C initialization produced a non-finite value"))
  all(isfinite, state.z) ||
    throw(ArgumentError("coupled z initialization produced a non-finite value"))
  state
end

function _CoupledTerms(; observation::Real = 0.0,
                       pilot::Real = 0.0,
                       tie::Real = 0.0,
                       response_regularization::Real = 0.0,
                       combiner_regularization::Real = 0.0,
                       smoothness::Real = 0.0,
                       parity::Real = 0.0)
  values = Float64[
    observation,
    pilot,
    tie,
    response_regularization,
    combiner_regularization,
    smoothness,
    parity,
  ]
  _CoupledTerms(values..., sum(values))
end

function _CoupledScratch(problem::_CoupledProblem)
  nparts, nc = size(problem.observations)
  max_check_degree = maximum((length(check) for check in problem.parity_sets); init = 0)
  _CoupledScratch(
    zeros(ComplexF64, nc),
    zeros(ComplexF64, nparts, nc),
    zeros(ComplexF64, nc),
    zeros(Float64, problem.nbits),
    zeros(ComplexF64, nc),
    zeros(Float64, problem.nbits),
    zeros(Float64, max_check_degree),
    zeros(Float64, max_check_degree),
  )
end

function _validate_coupled_objective_shapes(problem::_CoupledProblem,
                                            state::_CoupledState,
                                            scratch::_CoupledScratch)
  nparts, nc = size(problem.observations)
  expected_W = (nparts, _coupled_combiner_groups(problem))
  expected_C = (
    nparts,
    length(_coupled_ici_offsets(problem.spec)),
    _coupled_response_groups(problem),
    problem.spec.ofdm_symbols_per_solve,
  )
  size(state.W) == expected_W || throw(DimensionMismatch("W must have shape $expected_W"))
  size(state.C) == expected_C || throw(DimensionMismatch("C must have shape $expected_C"))
  length(state.z) == problem.nbits ||
    throw(DimensionMismatch("z must have length $(problem.nbits)"))
  length(scratch.symbols) == nc ||
    throw(DimensionMismatch("symbol scratch must have length $nc"))
  size(scratch.predicted_observations) == (nparts, nc) ||
    throw(DimensionMismatch("observation scratch must have shape $((nparts, nc))"))
  length(scratch.combined_symbols) == nc ||
    throw(DimensionMismatch("combiner scratch must have length $nc"))
  length(scratch.relaxed_bits) == problem.nbits ||
    throw(DimensionMismatch("bit scratch must have length $(problem.nbits)"))
  length(scratch.symbol_gradient) == nc ||
    throw(DimensionMismatch("symbol-gradient scratch must have length $nc"))
  length(scratch.bit_gradient) == problem.nbits ||
    throw(DimensionMismatch("bit-gradient scratch must have length $(problem.nbits)"))
  max_check_degree = maximum((length(check) for check in problem.parity_sets); init = 0)
  length(scratch.parity_prefix) == max_check_degree ||
    throw(DimensionMismatch("parity-prefix scratch must have length $max_check_degree"))
  length(scratch.parity_clamped) == max_check_degree ||
    throw(DimensionMismatch("parity-value scratch must have length $max_check_degree"))
  nothing
end

function _validate_coupled_gradient_shapes(problem::_CoupledProblem,
                                           gradient::_CoupledGradient)
  nparts = size(problem.observations, 1)
  expected_W = (nparts, _coupled_combiner_groups(problem))
  expected_C = (
    nparts,
    length(_coupled_ici_offsets(problem.spec)),
    _coupled_response_groups(problem),
    problem.spec.ofdm_symbols_per_solve,
  )
  size(gradient.W) == expected_W ||
    throw(DimensionMismatch("W gradient must have shape $expected_W"))
  size(gradient.C) == expected_C ||
    throw(DimensionMismatch("C gradient must have shape $expected_C"))
  length(gradient.z) == problem.nbits ||
    throw(DimensionMismatch("z gradient must have length $(problem.nbits)"))
  nothing
end

function _coupled_symbols!(problem::_CoupledProblem,
                           state::_CoupledState,
                           scratch::_CoupledScratch)
  @inbounds for bit_idx in eachindex(scratch.relaxed_bits)
    scratch.relaxed_bits[bit_idx] = if problem.inner_pilot_mask[bit_idx]
      _pm(problem.inner_pilot_bits[bit_idx])
    else
      tanh(0.5 * state.z[bit_idx])
    end
  end

  fill!(scratch.symbols, 0.0 + 0.0im)
  invsqrt2 = inv(sqrt(2.0))
  @inbounds for k in problem.active
    slot = problem.symbol_slots[k]
    if slot == 0
      scratch.symbols[k] = problem.fixed_symbols[k]
    else
      bit_i = 2slot - 1
      bit_q = bit_i + 1
      xi = scratch.relaxed_bits[bit_i]
      xq = bit_q <= problem.nbits ? scratch.relaxed_bits[bit_q] : 0.0
      scratch.symbols[k] = ComplexF64(xi, xq) * invsqrt2
    end
  end
  scratch.symbols
end

function _coupled_forward_and_combine!(problem::_CoupledProblem,
                                       state::_CoupledState,
                                       scratch::_CoupledScratch)
  fill!(scratch.predicted_observations, 0.0 + 0.0im)
  fill!(scratch.combined_symbols, 0.0 + 0.0im)
  nparts = size(problem.observations, 1)

  @inbounds for k in problem.active
    combiner_group = _coupled_combiner_group(problem, k)
    response_group = _coupled_response_group(problem, k)
    combined = 0.0 + 0.0im
    for branch in 1:nparts
      combined += conj(state.W[branch, combiner_group]) * problem.observations[branch, k]
    end
    scratch.combined_symbols[k] = combined

    for offset_pos in axes(problem.neighbor_idx, 1)
      q = problem.neighbor_idx[offset_pos, k]
      q == 0 && continue
      symbol = scratch.symbols[q]
      for branch in 1:nparts
        scratch.predicted_observations[branch, k] +=
          state.C[branch, offset_pos, response_group, 1] * symbol
      end
    end
  end
  nothing
end

# Scalar constrained counterpart of eq:juna-total. Every family is normalized
# by its own number of scalar contributions before its explicit weight is
# applied; this keeps coefficient meaning independent of miniature dimensions.
function _coupled_objective(problem::_CoupledProblem,
                            state::_CoupledState;
                            weights::_CoupledWeights = _COUPLED_UNIT_WEIGHTS,
                            scratch::_CoupledScratch = _CoupledScratch(problem))
  _validate_coupled_objective_shapes(problem, state, scratch)
  _coupled_symbols!(problem, state, scratch)
  _coupled_forward_and_combine!(problem, state, scratch)

  nparts = size(problem.observations, 1)
  observation_sum = 0.0
  @inbounds for k in problem.active, branch in 1:nparts
    residual = problem.observations[branch, k] -
               scratch.predicted_observations[branch, k]
    observation_sum += abs2(residual)
  end
  observation = weights.observation * observation_sum /
                (nparts * length(problem.active))

  pilot_sum = 0.0
  @inbounds for (pilot_pos, k) in enumerate(problem.pilot_idx)
    pilot_sum += abs2(scratch.combined_symbols[k] - problem.pilot_syms[pilot_pos])
  end
  pilot = isempty(problem.pilot_idx) ? 0.0 :
          weights.pilot * pilot_sum / length(problem.pilot_idx)

  tie_sum = 0.0
  dynamic_count = 0
  @inbounds for k in problem.data_idx
    problem.symbol_slots[k] == 0 && continue
    tie_sum += abs2(scratch.combined_symbols[k] - scratch.symbols[k])
    dynamic_count += 1
  end
  tie = dynamic_count == 0 ? 0.0 : weights.tie * tie_sum / dynamic_count

  response_regularization = weights.response_regularization *
                            sum(abs2, state.C) / length(state.C)
  combiner_regularization = weights.combiner_regularization *
                            sum(abs2, state.W) / length(state.W)

  smoothness_sum = 0.0
  smoothness_count = nparts * max(size(state.W, 2) - 1, 0)
  @inbounds for band_id in 1:size(state.W, 2)-1, branch in 1:nparts
    smoothness_sum += abs2(state.W[branch, band_id + 1] - state.W[branch, band_id])
  end
  smoothness = smoothness_count == 0 ? 0.0 :
               weights.smoothness * smoothness_sum / smoothness_count

  parity_sum = 0.0
  @inbounds for check in problem.parity_sets
    product = 1.0
    for bit_idx in check
      product *= scratch.relaxed_bits[bit_idx]
    end
    parity_sum += (1.0 - product)^2
  end
  parity = isempty(problem.parity_sets) ? 0.0 :
           weights.parity * parity_sum / length(problem.parity_sets)

  _CoupledTerms(
    observation = observation,
    pilot = pilot,
    tie = tie,
    response_regularization = response_regularization,
    combiner_regularization = combiner_regularization,
    smoothness = smoothness,
    parity = parity,
  )
end

# Analytic gradient of the constrained scalar objective. Complex entries use
# the same Wirtinger convention as _wz_loss_and_grad!: for g = dL/dconj(x),
# centered finite differences satisfy dL/dRe(x)=2real(g) and
# dL/dIm(x)=2imag(g). The z gradient is an ordinary real derivative.
function _coupled_objective_and_gradient!(gradient::_CoupledGradient,
                                          problem::_CoupledProblem,
                                          state::_CoupledState;
                                          weights::_CoupledWeights = _COUPLED_UNIT_WEIGHTS,
                                          scratch::_CoupledScratch = _CoupledScratch(problem))
  _validate_coupled_gradient_shapes(problem, gradient)
  terms = _coupled_objective(problem, state; weights = weights, scratch = scratch)

  fill!(gradient.W, 0.0 + 0.0im)
  fill!(gradient.C, 0.0 + 0.0im)
  fill!(gradient.z, 0.0)
  fill!(scratch.symbol_gradient, 0.0 + 0.0im)
  fill!(scratch.bit_gradient, 0.0)

  nparts = size(problem.observations, 1)
  observation_scale = weights.observation / (nparts * length(problem.active))
  if observation_scale > 0.0
    @inbounds for k in problem.active
      response_group = _coupled_response_group(problem, k)
      for offset_pos in axes(problem.neighbor_idx, 1)
        q = problem.neighbor_idx[offset_pos, k]
        q == 0 && continue
        symbol = scratch.symbols[q]
        for branch in 1:nparts
          residual = scratch.predicted_observations[branch, k] -
                     problem.observations[branch, k]
          response = state.C[branch, offset_pos, response_group, 1]
          gradient.C[branch, offset_pos, response_group, 1] +=
            observation_scale * residual * conj(symbol)
          scratch.symbol_gradient[q] +=
            observation_scale * conj(response) * residual
        end
      end
    end
  end

  if weights.pilot > 0.0 && !isempty(problem.pilot_idx)
    pilot_scale = weights.pilot / length(problem.pilot_idx)
    @inbounds for (pilot_pos, k) in enumerate(problem.pilot_idx)
      combiner_group = _coupled_combiner_group(problem, k)
      residual = scratch.combined_symbols[k] - problem.pilot_syms[pilot_pos]
      for branch in 1:nparts
        gradient.W[branch, combiner_group] +=
          pilot_scale * problem.observations[branch, k] * conj(residual)
      end
    end
  end

  dynamic_count = count(k -> problem.symbol_slots[k] != 0, problem.data_idx)
  if weights.tie > 0.0 && dynamic_count > 0
    tie_scale = weights.tie / dynamic_count
    @inbounds for k in problem.data_idx
      problem.symbol_slots[k] == 0 && continue
      combiner_group = _coupled_combiner_group(problem, k)
      residual = scratch.combined_symbols[k] - scratch.symbols[k]
      for branch in 1:nparts
        gradient.W[branch, combiner_group] +=
          tie_scale * problem.observations[branch, k] * conj(residual)
      end
      scratch.symbol_gradient[k] -= tie_scale * residual
    end
  end

  response_scale = weights.response_regularization / length(state.C)
  @inbounds for index in eachindex(state.C)
    gradient.C[index] += response_scale * state.C[index]
  end

  combiner_scale = weights.combiner_regularization / length(state.W)
  @inbounds for index in eachindex(state.W)
    gradient.W[index] += combiner_scale * state.W[index]
  end

  smoothness_count = nparts * max(size(state.W, 2) - 1, 0)
  if weights.smoothness > 0.0 && smoothness_count > 0
    smoothness_scale = weights.smoothness / smoothness_count
    @inbounds for band_id in 1:size(state.W, 2)-1, branch in 1:nparts
      difference = state.W[branch, band_id + 1] - state.W[branch, band_id]
      gradient.W[branch, band_id] -= smoothness_scale * difference
      gradient.W[branch, band_id + 1] += smoothness_scale * difference
    end
  end

  invsqrt2 = inv(sqrt(2.0))
  @inbounds for k in problem.data_idx
    slot = problem.symbol_slots[k]
    slot == 0 && continue
    bit_i = 2slot - 1
    bit_q = bit_i + 1
    symbol_gradient = scratch.symbol_gradient[k]
    scratch.bit_gradient[bit_i] += 2.0 * real(symbol_gradient) * invsqrt2
    if bit_q <= problem.nbits
      scratch.bit_gradient[bit_q] += 2.0 * imag(symbol_gradient) * invsqrt2
    end
  end

  if weights.parity > 0.0 && !isempty(problem.parity_sets)
    parity_scale = weights.parity / length(problem.parity_sets)
    @inbounds for check in problem.parity_sets
      product = 1.0
      for check_pos in eachindex(check)
        bit = scratch.relaxed_bits[check[check_pos]]
        scratch.parity_prefix[check_pos] = product
        scratch.parity_clamped[check_pos] = bit
        product *= bit
      end

      coefficient = -2.0 * parity_scale * (1.0 - product)
      suffix = 1.0
      for check_pos in length(check):-1:1
        bit_idx = check[check_pos]
        scratch.bit_gradient[bit_idx] +=
          coefficient * scratch.parity_prefix[check_pos] * suffix
        suffix *= scratch.parity_clamped[check_pos]
      end
    end
  end

  @inbounds for bit_idx in eachindex(state.z)
    problem.inner_pilot_mask[bit_idx] && continue
    relaxed = scratch.relaxed_bits[bit_idx]
    gradient.z[bit_idx] =
      scratch.bit_gradient[bit_idx] * 0.5 * (1.0 - relaxed * relaxed)
  end
  terms
end

_copy_coupled_state(state::_CoupledState) =
  _CoupledState(copy(state.W), copy(state.C), copy(state.z))

function _coupled_solve_workspace(problem::_CoupledProblem,
                                  initial::_CoupledState)
  current = _copy_coupled_state(initial)
  scratch = _CoupledScratch(problem)
  gradient = _CoupledGradient(problem)
  _validate_coupled_objective_shapes(problem, current, scratch)
  current, scratch, gradient
end

@inline function _clip_complex(value::ComplexF64, limit::Float64)
  magnitude = abs(value)
  magnitude > limit ? value * (limit / magnitude) : value
end

function _coupled_bcd_trial(problem::_CoupledProblem,
                            state::_CoupledState,
                            gradient::_CoupledGradient,
                            block::Symbol,
                            step::Float64,
                            config::_CoupledBCDConfig)
  trial = _copy_coupled_state(state)
  if block === :W
    @inbounds for i in eachindex(trial.W)
      trial.W[i] = _clip_complex(
        state.W[i] - step * gradient.W[i], config.complex_value_clip,
      )
    end
  elseif block === :C
    @inbounds for i in eachindex(trial.C)
      trial.C[i] = _clip_complex(
        state.C[i] - step * gradient.C[i], config.complex_value_clip,
      )
    end
  elseif block === :z
    @inbounds for i in eachindex(trial.z)
      problem.inner_pilot_mask[i] && continue
      trial.z[i] = clamp(
        state.z[i] - step * gradient.z[i], -config.logit_clip, config.logit_clip,
      )
    end
  else
    throw(ArgumentError("unknown coupled BCD block $block"))
  end
  trial
end

function _coupled_bcd_block_step(problem::_CoupledProblem,
                                 state::_CoupledState,
                                 loss::Float64,
                                 gradient::_CoupledGradient,
                                 block::Symbol,
                                 initial_step::Float64,
                                 weights::_CoupledWeights,
                                 config::_CoupledBCDConfig,
                                 scratch::_CoupledScratch)
  step = initial_step
  while step >= config.min_step
    trial = _coupled_bcd_trial(problem, state, gradient, block, step, config)
    trial_loss = _coupled_objective(
      problem, trial; weights = weights, scratch = scratch,
    ).total
    if isfinite(trial_loss) && trial_loss <= loss
      return trial, trial_loss
    end
    step *= config.shrink
  end
  state, loss
end

# Alternating safeguarded descent on the exact same frozen WCz objective used
# by the joint Adam solver. Each block gets a fresh hand-derived gradient and a
# backtracking line search; rejected trials leave the current state untouched.
function _coupled_bcd_solve(problem::_CoupledProblem,
                            initial::_CoupledState;
                            weights::_CoupledWeights = _COUPLED_RUNTIME_WEIGHTS,
                            config::_CoupledBCDConfig = _CoupledBCDConfig())
  _validate_coupled_bcd_config(config)
  current, scratch, gradient = _coupled_solve_workspace(problem, initial)
  initial_loss = _coupled_objective(
    problem, current; weights = weights, scratch = scratch,
  ).total
  isfinite(initial_loss) ||
    throw(ArgumentError("coupled BCD requires a finite initial objective"))

  loss = initial_loss
  best_state = _copy_coupled_state(current)
  selected_iter = 0
  loss_history = Float64[initial_loss]
  block_steps = ((:W, config.step_W), (:C, config.step_C), (:z, config.step_z))

  for cycle in 1:config.cycles
    for (block, initial_step) in block_steps
      terms = _coupled_objective_and_gradient!(
        gradient, problem, current; weights = weights, scratch = scratch,
      )
      loss = terms.total
      current, loss = _coupled_bcd_block_step(
        problem, current, loss, gradient, block, initial_step,
        weights, config, scratch,
      )
    end
    push!(loss_history, loss)
    if loss < loss_history[1]
      best_state = _copy_coupled_state(current)
      selected_iter = cycle
    end
  end

  _CoupledSolveResult(best_state, initial_loss, loss, loss_history, selected_iter)
end

# Joint bounded Adam over the hand-derived Wirtinger C/W gradients and ordinary
# real z gradient. The return value is always the best finite checkpoint seen,
# including iteration zero, so optimization cannot worsen its scalar objective.
function _coupled_wcz_solve(problem::_CoupledProblem,
                            initial::_CoupledState;
                            weights::_CoupledWeights = _COUPLED_RUNTIME_WEIGHTS,
                            config::_CoupledOptimizerConfig = _CoupledOptimizerConfig())
  _validate_coupled_optimizer_config(config)
  current, scratch, gradient = _coupled_solve_workspace(problem, initial)

  mW = zero(current.W)
  mC = zero(current.C)
  mz = zeros(Float64, length(current.z))
  vW = zeros(Float64, size(current.W))
  vC = zeros(Float64, size(current.C))
  vz = zeros(Float64, length(current.z))

  initial_terms = _coupled_objective_and_gradient!(
    gradient, problem, current; weights = weights, scratch = scratch,
  )
  isfinite(initial_terms.total) ||
    throw(ArgumentError("coupled optimizer requires a finite initial objective"))
  initial_loss = initial_terms.total
  best_loss = initial_loss
  best_state = _copy_coupled_state(current)
  selected_iter = 0
  loss_history = Float64[initial_loss]
  clamped_z = copy(current.z)

  for iteration in 1:config.steps
    _adam_step_complex!(
      current.W, mW, vW, gradient.W, iteration,
      config.alpha_W, config.beta1, config.beta2, config.epsilon,
      config.gradient_clip, config.complex_value_clip,
    )
    _adam_step_complex!(
      current.C, mC, vC, gradient.C, iteration,
      config.alpha_C, config.beta1, config.beta2, config.epsilon,
      config.gradient_clip, config.complex_value_clip,
    )
    _adam_step_real!(
      current.z, mz, vz, gradient.z, iteration,
      config.alpha_z, config.beta1, config.beta2, config.epsilon,
      config.gradient_clip, config.logit_clip,
    )
    @inbounds for bit_idx in eachindex(current.z)
      problem.inner_pilot_mask[bit_idx] || continue
      current.z[bit_idx] = clamped_z[bit_idx]
    end

    terms = _coupled_objective_and_gradient!(
      gradient, problem, current; weights = weights, scratch = scratch,
    )
    loss = terms.total
    push!(loss_history, loss)
    if isfinite(loss) && loss < best_loss
      best_loss = loss
      best_state = _copy_coupled_state(current)
      selected_iter = iteration
    end
    isfinite(loss) || break
  end

  _CoupledSolveResult(best_state, initial_loss, best_loss, loss_history, selected_iter)
end

# Map the optimized combiner and latent logits back to the established decoder
# boundary. C shapes the physical objective during optimization; W produces the
# equalized carrier vector and -z follows the package's positive-metric-is-bit-1
# convention before normalized min-sum BP scores the candidate.
function _coupled_state_candidate(m::Modulation,
                                  code::_Code,
                                  layout::_Layout,
                                  problem::_CoupledProblem,
                                  state::_CoupledState)
  problem.nbits == code.n ||
    throw(DimensionMismatch("coupled state bits must match the LDPC code length"))
  problem.active == layout.active ||
    throw(ArgumentError("coupled problem active carriers do not match the layout"))

  equalized = zeros(ComplexF64, Int(m.nc))
  @inbounds for k in problem.active
    combiner_group = _coupled_combiner_group(problem, k)
    for branch in axes(problem.observations, 1)
      equalized[k] +=
        conj(state.W[branch, combiner_group]) * problem.observations[branch, k]
    end
  end
  metrics = clamp.(-state.z, -_LLR_CLIP, _LLR_CLIP)
  _candidate_from_equalized(m, code, layout, equalized, metrics)
end

function _coupled_candidate(m::Modulation,
                            code::_Code,
                            layout::_Layout,
                            yparts,
                            seed;
                            weights::_CoupledWeights = _COUPLED_RUNTIME_WEIGHTS,
                            config::_CoupledOptimizerConfig = _COUPLED_PUBLIC_CONFIG)
  problem = _coupled_problem_from_receiver(m, code, layout, yparts)
  initial = _initial_coupled_state(m, code, layout, problem, seed)
  solved = _coupled_wcz_solve(problem, initial; weights = weights, config = config)
  _coupled_state_candidate(m, code, layout, problem, solved.state)
end

# Decoder-facing non-regression gate: the coupled optimizer may lower its
# scalar objective without producing a better LDPC candidate, so retain the
# seed unless the shared validity/syndrome/score ordering accepts the result.
function _juna_wcz_candidate(m::Modulation,
                             code::_Code,
                             layout::_Layout,
                             yparts,
                             seed=nothing;
                             weights::_CoupledWeights = _COUPLED_RUNTIME_WEIGHTS,
                             config::_CoupledOptimizerConfig = _COUPLED_PUBLIC_CONFIG)
  seed === nothing && (seed = _seed_candidate(m, code, layout, yparts))
  candidate = _coupled_candidate(
    m, code, layout, yparts, seed; weights = weights, config = config,
  )
  _juna_better(seed, candidate) ? candidate : seed
end
