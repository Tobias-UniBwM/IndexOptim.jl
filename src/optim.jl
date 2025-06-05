using Optimization, Combinatorics, ProgressMeter

import BlackBoxOptim

export 
    AbstractOptimizedObjective,
    AbstractIndexOptimizer

export 
    IndexOptimizationObjective,
    PreAllocatedOptimizationObjective,
    RegularizedOptimizationObjective,
    BlackBoxOptimIndexOptimizer,
    SciMLIndexOptimizer,
    PermutationIndexOptimizer

export 
    convert_dataset_for_index_optimization_objective,
    optimize_index

abstract type AbstractOptimizedObjective <: AbstractOptimized end

@mustimplement compute_fitness(objective::AbstractOptimizedObjective, workspace, parameters::AbstractArray{<:Real})

compute_fitness(objective::AbstractOptimizedObjective, parameters::AbstractArray{<:Real}) = compute_fitness(objective, create_workspace(objective), parameters)

(o::AbstractOptimizedObjective)(parameters::AbstractArray{<:Real}) = compute_fitness(o, parameters)
(o::AbstractOptimizedObjective)(workspace, parameters::AbstractArray{<:Real}) = compute_fitness(o, workspace, parameters)

struct IndexOptimizationObjective{TIndex <: AbstractOptimizedIndex, TImages <: AbstractArray{<:AbstractArray{<:AbstractArray{<:Real}}}, TMasks <: AbstractArray{<:AbstractArray{<:AbstractArray{<:Bool}}}, TWeights <: AbstractArray{<:AbstractArray{<:Real}}, TDescriptor <: AbstractIndexFitnessDescriptor} <: AbstractOptimizedObjective
    index::TIndex

    images::TImages
    masks::TMasks
    weights::TWeights

    descriptor::TDescriptor
end

function get_parameters_count(objective::IndexOptimizationObjective, inputs...)
    get_parameters_count(objective.index, length(first(objective.images)))
end

function create_workspace(objective::IndexOptimizationObjective, inputs...)
    (
        index_image = similar(first(first(objective.images))),
        index_workspace = create_workspace(objective.index, first(objective.images)),
        descriptor_workspace = create_workspace(objective.descriptor, first(first(objective.images)), first(first(objective.masks)))
    )
end

function compute_fitness(objective::IndexOptimizationObjective, workspace, parameters::AbstractArray{<:Real})
    fitness = zero(eltype(parameters))

    images = objective.images
    masks = objective.masks
    index = objective.index
    weights = objective.weights
    descriptor = objective.descriptor

    index_image, workspace_index, workspace_descriptor = workspace

    for (capture_images, capture_masks, capture_weights) in zip(images, masks, weights)
        compute_index!(index, workspace_index, capture_images, parameters, index_image)

        for (capture_mask, capture_weight) in zip(capture_masks, capture_weights)
            fitness += compute_fitness(descriptor, workspace_descriptor, index_image, capture_mask) * capture_weight
        end
    end

    fitness /= sum(ms -> length(ms), masks, init = zero(fitness))
end

function convert_dataset_for_index_optimization_objective(::Type{TData}, dataset::AbstractDataset) where TData <: Real
    names = Vector{String}()
    images = Vector{Vector{Matrix{TData}}}()
    masks = Vector{Vector{BitMatrix}}()
    weights = Vector{Vector{Float64}}()

    _category_weights = category_weights(dataset)

    for (c, a) in dataset
        capture_channels = channels(c)
        
        capture_images = Vector{Matrix{TData}}()

        for channel_name in sort!(collect(keys(capture_channels)))
            channel_image = capture_channels[channel_name]
            channel_image_raw = convert.(TData, channelview(channel_image))

            if length(size(channel_image)) > 2
                for channel_index in axes(channel_image, 1)
                    push!(capture_images, channel_image_raw[channel_index, :, :])
                end
            else
                push!(capture_images, channel_image_raw)
            end
        end

        capture_masks = [label_mask(a) for a in a]
        capture_weights = [_category_weights[category_name(a)] for a in a]

        if !isempty(capture_masks) && !isempty(capture_images)
            push!(images, capture_images)
            push!(masks, capture_masks)
            push!(weights, capture_weights)
            push!(names, name(c))
        end
    end

    return (names = names, images = images, masks = masks, weights = weights)
end

convert_dataset_for_index_optimization_objective(dataset::AbstractDataset) = convert_dataset_for_index_optimization_objective(Float64, dataset)

struct PreAllocatedOptimizationObjective{TObjective <: AbstractOptimizedObjective, TWorkspace} <: AbstractOptimizedObjective
    objective::TObjective
    workspace::TWorkspace
end

PreAllocatedOptimizationObjective(objective::AbstractOptimizedObjective) = PreAllocatedOptimizationObjective(objective, create_workspace(objective))

create_workspace(objective::PreAllocatedOptimizationObjective, inputs...) = objective.workspace
get_parameters_count(objective::PreAllocatedOptimizationObjective, inputs...) = get_parameters_count(objective.objective, inputs...)
compute_fitness(objective::PreAllocatedOptimizationObjective, workspace, parameters::AbstractArray{<:Real}) = compute_fitness(objective.objective, workspace, parameters)

struct RegularizedOptimizationObjective{TObjective <: AbstractOptimizedObjective, TNorm, TWeight <: Real} <: AbstractOptimizedObjective
    objective::TObjective
    norm::TNorm
    weight::TWeight
    normalize::Bool
end

create_workspace(objective::RegularizedOptimizationObjective, inputs...) = create_workspace(objective.objective, inputs...)
get_parameters_count(objective::RegularizedOptimizationObjective, inputs...) = get_parameters_count(objective.objective, inputs...)

function compute_fitness(objective::RegularizedOptimizationObjective, workspace, parameters::AbstractArray{<:Real}) 
    fitness = compute_fitness(objective.objective, workspace, parameters) 
    regularization = objective.norm(parameters)

    if objective.normalize
        regularization / length(parameters)
    end
    
    return fitness + regularization
end

abstract type AbstractIndexOptimizer end

@mustimplement optimize_index(
    optimizer::AbstractIndexOptimizer,
    index::AbstractOptimizedIndex,
    descriptor::AbstractIndexFitnessDescriptor,
    train_images::AbstractArray{<:AbstractArray{<:AbstractArray{<:Real}}},
    train_masks::AbstractArray{<:AbstractArray{<:AbstractArray{<:Bool}}},
    train_weights::AbstractArray{<:AbstractArray{<:Real}};
    x0::Optional{AbstractVector{<:Real}} = nothing,
    search_range::Optional{Union{AbstractVector{<:Real}, AbstractVector{<:AbstractVector{<:Real}}}} = nothing,
    valid_images::Optional{AbstractArray{<:AbstractArray{<:AbstractArray{<:Real}}}} = nothing,
    valid_masks::Optional{AbstractArray{<:AbstractArray{<:AbstractArray{<:Bool}}}} = nothing,
    valid_weights::Optional{AbstractArray{<:AbstractArray{<:Real}}} = nothing,
    log_train_progress = nothing,
    log_valid_progress = nothing
)

@kwdef struct BlackBoxOptimIndexOptimizer <: AbstractIndexOptimizer
    train_fitness_log_interval::Float64 = 5.
    valid_fitness_log_interval::Float64 = 60.
    bboptimize_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}()
end

function optimize_index(
    optimizer::BlackBoxOptimIndexOptimizer,
    index::AbstractOptimizedIndex,
    descriptor::AbstractIndexFitnessDescriptor, 
    train_images::AbstractArray{<:AbstractArray{<:AbstractArray{<:Real}}}, 
    train_masks::AbstractArray{<:AbstractArray{<:AbstractArray{<:Bool}}},
    train_weights::AbstractArray{<:AbstractArray{<:Real}};
    x0::Optional{AbstractVector{<:Real}} = nothing,
    search_range::Optional{Union{AbstractVector{<:Real}, AbstractVector{<:AbstractVector{<:Real}}}} = nothing,
    valid_images::Optional{AbstractArray{<:AbstractArray{<:AbstractArray{<:Real}}}} = nothing,
    valid_masks::Optional{AbstractArray{<:AbstractArray{<:AbstractArray{<:Bool}}}} = nothing,
    valid_weights::Optional{AbstractArray{<:AbstractArray{<:Real}}} = nothing,
    log_train_progress = nothing,
    log_valid_progress = nothing
)
    bboptimize_kwargs = optimizer.bboptimize_kwargs

    bboptimize_kwargs[:NumDimensions] = get_parameters_count(index, first(train_images))

    if !isnothing(search_range) && !isempty(search_range)
        bboptimize_kwargs[:SearchRange] = map(s -> Tuple(convert.(Float64, s)), search_range)     
    end

    is_multithreaded = haskey(bboptimize_kwargs, :NThreads) && Threads.nthreads() >= bboptimize_kwargs[:NThreads]

    training_objective = IndexOptimizationObjective(index, train_images, train_masks, train_weights, descriptor)

    if !is_multithreaded
        training_objective = PreAllocatedOptimizationObjective(training_objective)
    end

    if !isnothing(valid_images) && !isnothing(valid_masks)
        valid_objective = IndexOptimizationObjective(index, valid_images, valid_masks, valid_weights, descriptor)
        
        if !is_multithreaded
            valid_objective = PreAllocatedOptimizationObjective(valid_objective)
        end
    else
        valid_objective = nothing
    end

    last_train_log_ref = Ref(time())
    last_valid_log_ref = Ref(time())

    callback = function (oc, o = optimizer, train_time_ref = last_train_log_ref, valid_time_ref = last_valid_log_ref, valid_obj = valid_objective, log_train = log_train_progress, log_valid = log_valid_progress)
        current_time = time()

        if (current_time - train_time_ref[]) >= o.train_fitness_log_interval && !isnothing(log_train)
            log_train(BlackBoxOptim.best_fitness(oc), BlackBoxOptim.best_candidate(oc), BlackBoxOptim.num_steps(oc))
            train_time_ref[] = current_time
        end

        if (current_time - valid_time_ref[]) >= o.valid_fitness_log_interval && !isnothing(log_valid) && !isnothing(valid_obj)
            log_valid(valid_obj(BlackBoxOptim.best_candidate(oc)), BlackBoxOptim.best_candidate(oc), BlackBoxOptim.num_steps(oc))
            valid_time_ref[] = current_time
        end
    end

    if !isnothing(log_train_progress) || !isnothing(log_valid_progress)
        bboptimize_kwargs[:CallbackFunction] = callback
        bboptimize_kwargs[:CallbackInterval] = min(optimizer.train_fitness_log_interval, optimizer.valid_fitness_log_interval)
    end

    if isnothing(x0) || isempty(x0)
        results = BlackBoxOptim.bboptimize(training_objective; bboptimize_kwargs...)
    else
        results = BlackBoxOptim.bboptimize(training_objective, x0; bboptimize_kwargs...)
    end

    fitness = BlackBoxOptim.best_fitness(results)
    candidate = BlackBoxOptim.best_candidate(results)

    return (fitness = fitness, candidate = candidate, termination_reason = BlackBoxOptim.stop_reason(results))
end

struct SciMLIndexOptimizer{TSolver} <: AbstractIndexOptimizer
    solver::TSolver
    problem_kwargs::Dict{Symbol, Any}
    verbose::Bool
    train_fitness_log_interval::Float64
    valid_fitness_log_interval::Float64
end

function optimize_index(
    optimizer::SciMLIndexOptimizer,
    index::AbstractOptimizedIndex,
    descriptor::AbstractIndexFitnessDescriptor, 
    train_images::AbstractArray{<:AbstractArray{<:AbstractArray{<:Real}}}, 
    train_masks::AbstractArray{<:AbstractArray{<:AbstractArray{<:Bool}}},
    train_weights::AbstractArray{<:AbstractArray{<:Real}};
    x0::Optional{AbstractVector{<:Real}} = nothing,
    search_range::Optional{Union{AbstractVector{<:Real}, AbstractVector{<:AbstractVector{<:Real}}}} = nothing,
    valid_images::Optional{AbstractArray{<:AbstractArray{<:AbstractArray{<:Real}}}} = nothing,
    valid_masks::Optional{AbstractArray{<:AbstractArray{<:AbstractArray{<:Bool}}}} = nothing,
    valid_weights::AbstractArray{<:AbstractArray{<:Real}},
    log_train_progress = nothing,
    log_valid_progress = nothing
)
    problem_kwargs = optimizer.problem_kwargs

    parameters_count = get_parameters_count(index, first(train_images))

    if !isnothing(search_range) && !isempty(search_range)
        problem_kwargs[:lb] = [first(t) for t in search_range]
        problem_kwargs[:ub] = [last(t) for t in search_range]
    end

    if isnothing(x0)
        x0 = Float64[]
    end

    if isempty(x0)
        resize!(x0, parameters_count)

        if !isnothing(search_range) && !isempty(search_range)
            for (ix, is) in zip(eachindex(x0), eachindex(search_range))
                x0[ix] = first(search_range[is]) + rand() * (last(search_range[is]) - first(search_range[is]))
            end
        else
            x0 .= randn(eltype(x0), parameters_count)
        end
    end

    train_objective = PreAllocatedOptimizationObjective(IndexOptimizationObjective(index, train_images, train_masks, train_weights, descriptor))

    if !isnothing(valid_images) && !isnothing(valid_masks)
        valid_objective = PreAllocatedOptimizationObjective(IndexOptimizationObjective(index, valid_images, valid_masks, valid_weights, descriptor))
    else
        valid_objective = nothing
    end

    start_time_ref = Ref(time())

    last_train_log_ref = Ref(start_time_ref[])
    last_valid_log_ref = Ref(last_train_log_ref[])

    if !isnothing(log_train_progress) || !isnothing(log_valid_progress)
        callback = function (state, fitness, valid_objective = valid_objective, start_time_ref = start_time_ref, last_train_log_ref = last_train_log_ref, train_log_interval = optimizer.train_fitness_log_interval, last_valid_log_ref = last_valid_log_ref, valid_log_interval = optimizer.valid_fitness_log_interval, verbose = optimizer.verbose)
            current_time = time()
            seconds_passed = convert(Int, round(current_time - start_time_ref[]))
            
            step = state.iter
            candidate = state.u

            # TODO: remove as soon as that has been fixed
            if iszero(step)
                step = seconds_passed
            end

            if (current_time - last_train_log_ref[]) >= train_log_interval
                !isnothing(log_train_progress) && log_train_progress(fitness, candidate, step)
                verbose && println("Seconds Passed: $seconds_passed - Iteration: $step - Fitness: $fitness")

                last_train_log_ref[] = current_time
            end

            if !isnothing(log_valid_progress) && !isnothing(valid_objective) && (current_time - last_valid_log_ref[]) >= valid_log_interval
                log_valid_progress(valid_objective(candidate), candidate, step)
                last_valid_log_ref[] = current_time
            end

            return false
        end

        problem_kwargs[:callback] = callback
    end

    fitness_function = (parameters, _p, train_objective = train_objective) -> train_objective(parameters)

    optimization_problem = OptimizationProblem(fitness_function, x0; problem_kwargs...)

    result = solve(optimization_problem, optimizer.solver)

    success = Optimization.SciMLBase.successful_retcode(result.retcode)
    
    return (fitness = result.objective, candidate = result.u, termination_reason = "Success: $success - Code: $(string(result.retcode))")
end

struct PermutationIndexOptimizer <: AbstractIndexOptimizer
    indices_groups::Vector{Vector{Vector{Int}}}
    log_all_candidates::Bool
end

function optimize_index(
    optimizer::PermutationIndexOptimizer,
    index::AbstractOptimizedIndex,
    descriptor::AbstractIndexFitnessDescriptor, 
    train_images::AbstractArray{<:AbstractArray{<:AbstractArray{<:Real}}}, 
    train_masks::AbstractArray{<:AbstractArray{<:AbstractArray{<:Bool}}},
    train_weights::AbstractArray{<:AbstractArray{<:Real}};
    x0::Optional{AbstractVector{<:Real}} = nothing,
    search_range::Optional{Union{AbstractVector{<:Real}, AbstractVector{<:AbstractVector{<:Real}}}} = nothing,
    valid_images::Optional{AbstractArray{<:AbstractArray{<:AbstractArray{<:Real}}}} = nothing,
    valid_masks::Optional{AbstractArray{<:AbstractArray{<:AbstractArray{<:Bool}}}} = nothing,
    valid_weights::Optional{AbstractArray{<:AbstractArray{<:Real}}} = nothing,
    log_train_progress = nothing,
    log_valid_progress = nothing
)
    @assert !isnothing(x0) && !isempty(x0) "permutable initial candidate is required"
    @assert all(g -> all(i -> length(i) == length(first(g)), g), optimizer.indices_groups) "permutation indices of each group have to be of the same size"

    train_objective = PreAllocatedOptimizationObjective(IndexOptimizationObjective(index, train_images, train_masks, train_weights, descriptor))

    if !isnothing(valid_images) && !isnothing(valid_masks) && !isnothing(valid_weights)
        valid_objective = PreAllocatedOptimizationObjective(IndexOptimizationObjective(index, valid_images, valid_masks, valid_weights, descriptor))
    else
        valid_objective = nothing
    end

    indices_permutations = [permutations(eachindex(first(indices_group))) for indices_group in optimizer.indices_groups]
    indices_permutations_product = Iterators.product(indices_permutations...)
    past_candidates = Set{typeof(x0)}()
    
    best_train_fitness = floatmax(eltype(x0))
    best_train_candidate = similar(x0)
    best_valid_fitness = floatmax(eltype(x0))

    step = 1

    @showprogress desc="Searching optimal permutation..." for indices_group_permutations in indices_permutations_product
        candidate = copy(x0)

        for (indices_group, indices_permutation) in zip(optimizer.indices_groups, indices_group_permutations)
            for indices in indices_group
                for (i, ip) in zip(indices, indices_permutation)
                    candidate[i] = x0[indices[ip]]
                end
            end
        end

        if !(candidate in past_candidates)
            push!(past_candidates, candidate)

            train_fitness = train_objective(candidate)

            if train_fitness < best_train_fitness
                best_train_fitness = train_fitness
                best_train_candidate .= candidate
            end

            if !isnothing(log_train_progress)
                log_train_progress(train_fitness, candidate, step, optimizer.log_all_candidates)
            end

            if !isnothing(valid_objective)
                valid_fitness = valid_objective(candidate)

                if valid_fitness < best_valid_fitness
                    best_valid_fitness = valid_fitness
                end

                if !isnothing(log_valid_progress)
                    log_valid_progress(valid_fitness, candidate, step, optimizer.log_all_candidates)
                end
            end

            step += 1
        end
    end
    
    return (fitness = best_train_fitness, candidate = best_train_candidate, termination_reason = "permutation sequence exhausted")
end