using Images, Optimization, OptimizationPRIMA, LinearAlgebra, DataStructures, Logging, TensorBoardLogger

import YAML

export 
    register_parameter_modification_config_parser,
    register_optimizer_config_parser,
    register_fitness_descriptor_config_parser

export 
    process_index_config,
    process_optimized_index_config,
    process_parameter_modifications_config,
    process_regularization_config,
    process_optimizer_config,
    process_fitness_descriptor_config,
    process_results_config,
    process_datasets_config,
    execute_index_optimization_config,
    execute_index_evaluation_config

const DEFAULT_VALUE_EMPTY_CONFIG = Dict()

const KEY_TENSORBOARD_PREFIX = "indexoptim."
const KEY_TENSORBOARD_CANDIDATES = "candidates"
const KEY_TENSORBOARD_METRICS = "metrics"

const NAME_METRIC_FITNESS = "fitness"

const TAG_TRAIN = "train"
const TAG_VALID = "valid"
const TAG_TEST = "test"

const KEY_INDEX_CLEAN = "clean"
const KEY_INDEX_INDEX = "index"
const KEY_INDEX_INDICES = "indices"
const KEY_INDEX_REDUCTION = "reduction"
const KEY_INDEX_TRANSFORMS = "transforms"

function process_optimized_index_config(config::AbstractDict)
    if haskey(config, KEY_INDEX_CLEAN)
        index = CleanOptimizedIndex(process_optimized_index_config(config[KEY_INDEX_CLEAN]))
    elseif haskey(config, KEY_INDEX_INDEX)
        index = process_optimized_index_config(config[KEY_INDEX_INDEX])
    else
        reduction = eval(Symbol(config[KEY_INDEX_REDUCTION]))()

        if haskey(config, KEY_INDEX_TRANSFORMS)
            transforms = map(t -> eval(Symbol(t))(), config[KEY_INDEX_TRANSFORMS])
            transform = length(transforms) > 1 ? ChainTransform(transforms) : first(transforms)
        else
            transform = IdentityTransform()
        end

        if haskey(config, KEY_INDEX_INDICES)
            indices = map(c -> process_optimized_index_config(c[KEY_INDEX_INDEX]), config[KEY_INDEX_INDICES])
            index = OptimizedIndexCascade(indices, transform, reduction)
        else
            index = OptimizedIndex(transform, reduction)
        end
    end
    
    return index
end

function process_index_config(config::AbstractDict)
    process_optimized_index_config(config)
end

const KEY_ROUND_DIGITS = "digits"

function process_parameter_modification_round_config(config::AbstractDict)
    digits = config[KEY_ROUND_DIGITS]
    
    return RoundedParameters(digits)
end

const KEY_DROPOUT_COUNT = "count"
const KEY_DROPOUT_INDICES = "indices"

function process_parameter_modification_dropout_config(config::AbstractDict)
    dropout_count = config[KEY_DROPOUT_COUNT]
    parameter_indices = [convert.(Int, i) for i in config[KEY_DROPOUT_INDICES]]

    return MinSumDropoutParameters(parameter_indices, dropout_count)
end

const KEY_MODIFICATION_DROPOUT = "dropout"
const KEY_MODIFICATION_ROUND = "round"

const MAP_PARAMETER_MODIFICATIONS_CONFIG_PARSER = Dict{String, Any}(
    KEY_MODIFICATION_DROPOUT => process_parameter_modification_dropout_config,
    KEY_MODIFICATION_ROUND => process_parameter_modification_round_config
)

function register_parameter_modification_config_parser(name::AbstractString, parser)
    MAP_PARAMETER_MODIFICATIONS_CONFIG_PARSER[name] = parser
end

function process_parameter_modifications_config(config::AbstractDict)
    name, mod_config = first(config)
    MAP_PARAMETER_MODIFICATIONS_CONFIG_PARSER[name](mod_config)
end

process_parameter_modifications_config(config::AbstractVector) = ChainParameterModification([process_parameter_modifications_config(c) for c in config])

const KEY_PERM_INDICES = "indices"
const KEY_PERM_LOG_ALL = "log_all"

process_permutation_optimizer_config(config::AbstractDict) = PermutationIndexOptimizer(
    [[convert.(Int, idxs) for idxs in group] for group in config[KEY_PERM_INDICES]],
    get(config, KEY_PERM_LOG_ALL, false)
)

const KEY_BBOX_LOG_TRAIN_INTERVAL = "log_train"
const KEY_BBOX_LOG_VALID_INTERVAL = "log_valid"

const DEFAULT_VALUE_BBOX_LOG_TRAIN_INTERVAL = 5.
const DEFAULT_VALUE_BBOX_LOG_VALID_INTERVAL = 60.

function process_bbox_optimizer_config(config::AbstractDict)
    train_log_interval = pop!(config, KEY_BBOX_LOG_TRAIN_INTERVAL, DEFAULT_VALUE_BBOX_LOG_TRAIN_INTERVAL)
    valid_log_interval = pop!(config, KEY_BBOX_LOG_VALID_INTERVAL, DEFAULT_VALUE_BBOX_LOG_VALID_INTERVAL)

    bboptimize_kwargs = keys_as_symbols(config)

    return BlackBoxOptimIndexOptimizer(train_log_interval, valid_log_interval, bboptimize_kwargs)
end

function process_sciml_optimizer_solver_config(config::AbstractString)
    # TODO: don't use eval and do a proper parsing
    eval(Symbol(config))()
end

const KEY_SCIML_SOLVER = "solver"
const KEY_SCIML_PROBLEM_KWARGS = "problem_kwargs"
const KEY_SCIML_VERBOSE = "verbose"
const KEY_SCIML_LOG_TRAIN_INTERVAL = "log_train"
const KEY_SCIML_LOG_VALID_INTERVAL = "log_valid"

const DEFAULT_VALUE_SCIML_VERBOSE = true
const DEFAULT_VALUE_SCIML_LOG_TRAIN_INTERVAL = 5.
const DEFAULT_VALUE_SCIML_LOG_VALID_INTERVAL = 60.

function process_sciml_optimizer_config(config::AbstractDict)
    solver = process_sciml_optimizer_solver_config(config[KEY_SCIML_SOLVER])
    problem_kwargs = keys_as_symbols(get(config, KEY_SCIML_PROBLEM_KWARGS, DEFAULT_VALUE_EMPTY_CONFIG))
    verbose = get(config, KEY_SCIML_VERBOSE, DEFAULT_VALUE_SCIML_VERBOSE)
    log_train = get(config, KEY_SCIML_LOG_TRAIN_INTERVAL, DEFAULT_VALUE_SCIML_LOG_TRAIN_INTERVAL)
    log_valid = get(config, KEY_SCIML_LOG_VALID_INTERVAL, DEFAULT_VALUE_SCIML_LOG_VALID_INTERVAL)

    return SciMLIndexOptimizer(solver, problem_kwargs, verbose, log_train, log_valid)
end

const NAME_BBOX_OPTIMIZER_CONFIG_PARSER = "bbox"
const NAME_SCIML_OPTIMIZER_CONFIG_PARSER = "sciml"
const NAME_PERM_OPTIMIZER_CONFIG_PARSER = "perm"

const MAP_OPTIMIZER_CONFIG_PARSER = Dict{String, Any}(
    NAME_BBOX_OPTIMIZER_CONFIG_PARSER => process_bbox_optimizer_config,
    NAME_SCIML_OPTIMIZER_CONFIG_PARSER => process_sciml_optimizer_config,
    NAME_PERM_OPTIMIZER_CONFIG_PARSER => process_permutation_optimizer_config
)

function register_optimizer_config_parser(name::AbstractString, parser)
    MAP_OPTIMIZER_CONFIG_PARSER[name] = parser
end

const KEY_REGULARIZATION_NORM = "norm"
const KEY_REGULARIZATION_WEIGHT = "weight"
const KEY_REGULARIZATION_NORMALIZE = "normalize"

const DEFAULT_VALUE_REGULARIZATION_NORM = 2
const DEFAULT_VALUE_REGULARIZATION_WEIGHT = 1.
const DEFAULT_VALUE_REGULARIZATION_NORMALIZE = false

function process_regularization_config(config::AbstractDict, objective::AbstractOptimizedObjective)
    norm_p = get(config, KEY_REGULARIZATION_NORM, DEFAULT_VALUE_REGULARIZATION_NORM)
    weight = get(config, KEY_REGULARIZATION_WEIGHT, DEFAULT_VALUE_REGULARIZATION_WEIGHT)
    normalize = get(config, KEY_REGULARIZATION_NORMALIZE, DEFAULT_VALUE_REGULARIZATION_NORMALIZE)

    RegularizedOptimizationObjective(objective, (w, p = norm_p) -> norm(w, p), weight, normalize)    
end

const KEY_OPTIMIZER_X0 = "x0"
const KEY_OPTIMIZER_RANGE = "range"

const KEYS_OPTIMIZER = Set([
    KEY_OPTIMIZER_X0, 
    KEY_OPTIMIZER_RANGE
])

const DEFAULT_VALUE_OPTIMIZER_X0 = Float64[]
const DEFAULT_VALUE_OPTIMIZER_RANGE = Float64[]

function process_optimizer_config(config::AbstractDict, parameters_count::Integer)
    x0 = get(config, KEY_OPTIMIZER_X0, DEFAULT_VALUE_OPTIMIZER_X0)
    search_range = get(config, KEY_OPTIMIZER_RANGE, DEFAULT_VALUE_OPTIMIZER_RANGE)

    if !isempty(x0)
        x0 = convert.(Float64, x0)
        
        if parameters_count != length(x0)
            x0 = repeat(x0, parameters_count ÷ length(x0))
        end
    
        @assert length(x0) == parameters_count "Mismatch of starting point dimensions!"
    end

    if !isempty(search_range)
        if eltype(search_range) <: Real
            search_range = [search_range, ]
        end
    
        if !isempty(first(search_range))
            if parameters_count != length(search_range)
                search_range = repeat(search_range, parameters_count ÷ length(search_range))
            end
    
            @assert length(search_range) == parameters_count "Mismatch of search range dimensions!"
        end
    end

    config_keys = collect(keys(config))
    optimizer_idx = findfirst(k -> !(k in KEYS_OPTIMIZER), config_keys)
    optimizer_name = config_keys[optimizer_idx]

    x0, search_range, MAP_OPTIMIZER_CONFIG_PARSER[optimizer_name](something(config[optimizer_name], DEFAULT_VALUE_EMPTY_CONFIG))
end

function process_tvi_fitness_descriptor(::AbstractDict)
    return TargetVisibilityIndexFitnessDescriptor()
end

const NAME_TVI_DESCRIPTOR_CONFIG_PARSER = "tvi"

const MAP_FITNESS_DESCRIPTOR_CONFIG_PARSER = Dict{String, Any}(
    NAME_TVI_DESCRIPTOR_CONFIG_PARSER => process_tvi_fitness_descriptor,
)

function register_fitness_descriptor_config_parser(name::AbstractString, parser)
    MAP_FITNESS_DESCRIPTOR_CONFIG_PARSER[name] = parser
end

function process_fitness_descriptor_config(config::AbstractDict)
    name, parser_config = first(config)
    MAP_FITNESS_DESCRIPTOR_CONFIG_PARSER[name](something(parser_config, DEFAULT_VALUE_EMPTY_CONFIG))
end

const KEY_TENSORBOARD_PATH = "path"
const KEY_TENSORBOARD_MODE = "mode"

const VALUE_TENSORBOARD_MODE_APPEND = "append"
const VALUE_TENSORBOARD_MODE_INCREMENT = "increment"
const VALUE_TENSORBOARD_MODE_OVERWRITE = "overwrite"

const DEFAULT_VALUE_TENSORBOARD_PATH = "tensorboard"
const DEFAULT_VALUE_TENSORBOARD_MODE = VALUE_TENSORBOARD_MODE_INCREMENT

const MAP_TENSORBOARD_MODE = Dict(
    VALUE_TENSORBOARD_MODE_APPEND => tb_append,
    VALUE_TENSORBOARD_MODE_INCREMENT => tb_increment,
    VALUE_TENSORBOARD_MODE_OVERWRITE => tb_overwrite
)

function process_tensorboard_logger_config(config::AbstractDict)
    return (path = get(config, KEY_TENSORBOARD_PATH, DEFAULT_VALUE_TENSORBOARD_PATH), mode = get(config, KEY_TENSORBOARD_MODE, DEFAULT_VALUE_TENSORBOARD_MODE))
end

function process_tensorboard_logger_config(tensorboard_path::AbstractString)
    return (path = tensorboard_path, mode = DEFAULT_VALUE_TENSORBOARD_MODE)
end

const KEY_RESULTS_PATH = "path"
const KEY_RESULTS_STATS = "stats"
const KEY_RESULTS_IMAGES = "images"
const KEY_RESULTS_TENSORBOARD = "tensorboard"
const KEY_RESULTS_STEP = "step"

const DEFAULT_VALUE_RESULTS_PATH = "."
const DEFAULT_VALUE_RESULTS_STEP = 0

function process_results_config(config::AbstractDict)
    results_dir = get(config, KEY_RESULTS_PATH, DEFAULT_VALUE_RESULTS_PATH)

    if !isdir(results_dir)
        mkpath(results_dir)
    end

    images_path = get(config, KEY_RESULTS_IMAGES, "")
    
    if !isempty(images_path)
        images_path = joinpath(results_dir, images_path)

        if !isdir(images_path)
            mkpath(images_path)
        end
    end

    stats_path = get(config, KEY_RESULTS_STATS, "")

    if !isempty(stats_path)
        stats_path = joinpath(results_dir, stats_path)
    end

    if haskey(config, KEY_RESULTS_TENSORBOARD)
        tensorboard_path, tensorboard_mode = process_tensorboard_logger_config(something(config[KEY_RESULTS_TENSORBOARD], DEFAULT_VALUE_EMPTY_CONFIG))
    else
        tensorboard_path = ""
        tensorboard_mode = ""
    end

    if !isempty(tensorboard_path)
        tensorboard_path = joinpath(results_dir, tensorboard_path)
    end

    return (
        stats = stats_path, 
        images = images_path, 
        tensorboard_path = tensorboard_path, 
        tensorboard_mode = tensorboard_mode, 
        step = get(config, KEY_RESULTS_STEP, DEFAULT_VALUE_RESULTS_STEP)
    )
end

const KEY_DATASET_TRAIN = "train"
const KEY_DATASET_VALID = "valid"
const KEY_DATASET_TEST = "test"

function process_datasets_config(config::AbstractDict)
    dataset_train, dataset_valid, dataset_test = nothing, nothing, nothing
    
    if haskey(config, KEY_DATASET_TRAIN)
        dataset_train = process_dataset_config(config[KEY_DATASET_TRAIN])

        if haskey(config, KEY_DATASET_VALID)
            dataset_valid = process_dataset_config(config[KEY_DATASET_VALID])
        end

        if haskey(config, KEY_DATASET_TEST)
            dataset_test = process_dataset_config(config[KEY_DATASET_TEST])
        end
    else
        dataset_train = process_dataset_config(config)
    end

    return (dataset_train = dataset_train, dataset_valid = dataset_valid, dataset_test = dataset_test)
end

process_datasets_config(config::AbstractVector) = process_datasets_config(Dict(KEY_DATASET_TRAIN => config))

function log_optimizer_progress(logger::AbstractLogger, metrics::AbstractDict, step::Integer = 1)
    Base.with_logger(logger) do
        for (i, (metric_name, metric_value)) in enumerate(metrics)
            eval(:(@info KEY_TENSORBOARD_METRICS $(Symbol(metric_name)) = $(metric_value) step = $(step)))
        end 
    end
end

function log_optimizer_progress(logger::TBLogger, metrics::AbstractDict, step::Integer = 1)
    for (metric_name, metric_value) in metrics
        log_value(logger, "$(KEY_TENSORBOARD_METRICS)/$(metric_name)", metric_value, step = step)
    end
end

log_optimizer_progress(::Nothing, metrics::AbstractDict, step::Integer = 1) = log_optimizer_progress(global_logger(), metrics, step)
log_optimizer_progress(metrics::AbstractDict, step::Integer = 1) = log_optimizer_progress(global_logger(), metrics, step)

function log_stats(logger::AbstractLogger, stats::AbstractDict)
    stats_string = YAML.write(stats)

    with_logger(logger) do 
        @info "text" stats = stats_string
    end
end

function log_stats(logger::TBLogger, stats::AbstractDict)
    stats_string = YAML.write(stats)

    # patch string since line breaks don't work (https://github.com/JuliaLogging/TensorBoardLogger.jl/issues/145)
    stats_string = replace(stats_string, "\n" => "<br>", "\"" => "")

    log_text(logger, "stats", "<pre>$(stats_string)</pre>", step = 0)
end

log_stats(::Nothing, stats::AbstractDict) = log_stats(global_logger(), stats)
log_stats(stats::AbstractDict) = log_stats(global_logger(), stats)

function log_config(logger::AbstractLogger, config::AbstractDict)
    config_string = YAML.write(config)

    with_logger(logger) do 
        @info "text" config = config_string
    end
end

function log_config(logger::TBLogger, config::AbstractDict)
    config_string = YAML.write(config)
    
    # patch string since line breaks don't work (https://github.com/JuliaLogging/TensorBoardLogger.jl/issues/145)
    config_string = replace(config_string, "\n" => "<br>", "\"" => "")

    log_text(logger, "config", "<pre>$(config_string)</pre>", step = 0)
end

log_config(::Nothing, config::AbstractDict) = log_config(global_logger(), config)
log_config(config::AbstractDict) = log_config(global_logger(), config)

function log_candidate(logger::AbstractLogger, tag::AbstractString, fitness::Real, candidate::AbstractVector{<:Real}, step::Integer = 0)
    candidate_string = YAML.write(candidate)
    
    # patch string since line breaks don't work (https://github.com/JuliaLogging/TensorBoardLogger.jl/issues/145)
    candidate_string = replace(candidate_string, "\n" => "<br>", "\"" => "")

    with_logger(logger) do
        @info "text" tag = tag fitness = fitness candidate = candidate_string x = step
    end
end

function log_candidate(logger::TBLogger, tag::AbstractString, fitness::Real, candidate::AbstractVector{<:Real}, step::Integer = 0)
    candidate_string = YAML.write(OrderedDict("fitness" => fitness, "candidate" => candidate))
    
    # patch string since line breaks don't work (https://github.com/JuliaLogging/TensorBoardLogger.jl/issues/145)
    candidate_string = replace(candidate_string, "\n" => "<br>", "\"" => "")

    log_text(logger, "$(KEY_TENSORBOARD_CANDIDATES)/$(tag)", "<pre>$(candidate_string)</pre>", step = step)
end

log_candidate(::Nothing, tag::AbstractString, fitness::Real, candidate::AbstractVector{<:Real}, step = 0) = log_candidate(global_logger(), tag, fitness, candidate, step)
log_candidate(tag::AbstractString, fitness::Real, candidate::AbstractVector{<:Real}, step = 0) = log_candidate(global_logger(), tag, fitness, candidate, step)

function log_index_image(logger::TBLogger, name::AbstractString, image::AbstractMatrix)
    with_logger(logger) do 
        eval(:(@info "indices" $(Symbol(name)) = $(TBImage(image, HW)) log_step_increment = 0))
    end
end

function log_index_image(::AbstractLogger, ::AbstractString, ::AbstractMatrix)
    # don't log images to an arbitrary logger
    return
end

log_index_image(::Nothing, name::AbstractString, image::AbstractMatrix) = log_index_image(global_logger(), name, image)
log_index_image(name::AbstractString, image::AbstractMatrix) = log_index_image(global_logger(), name, image)

function log_index_images(directory::AbstractString, tag::AbstractString, logger::AbstractLogger, index::AbstractOptimizedIndex, images::AbstractArray{<:AbstractArray{<:AbstractArray{<:Real}}}, names::AbstractArray{<:AbstractString}, parameters::AbstractArray{<:Real})
    for (n, i) in zip(names, images)
        index_image = convert.(Gray, normalize_unit!(compute_index(index, i, parameters)))

        log_index_image(logger, joinpath(tag, n), index_image)

        if !isempty(directory)
            images_dir = joinpath(directory, tag)

            if !isdir(images_dir)
                mkpath(images_dir)
            end

            save(joinpath(images_dir, pathext(n, INDEX_IMAGE_FILE_EXTENSION)), index_image)
        end
    end
end

const KEY_INDEX_OPTIM_DATASET = "dataset"
const KEY_INDEX_OPTIM_INDEX = "index"
const KEY_INDEX_OPTIM_PARAMETER_MODIFICATIONS = "parameter_modifications"
const KEY_INDEX_OPTIM_OPTIMIZER = "optimizer"
const KEY_INDEX_OPTIM_DESCRIPTOR = "descriptor"
const KEY_INDEX_OPTIM_REGULARIZATION = "regularization"
const KEY_INDEX_OPTIM_RESULTS = "results"

const INDEX_IMAGE_FILE_EXTENSION = "png"

const KEY_STATS_PARAMETERS = "parameters_count"
const KEY_STATS_FITNESS_TRAIN = "best_fitness_train"
const KEY_STATS_FITNESS_VALID = "best_fitness_valid"
const KEY_STATS_FITNESS_TEST = "fitness_test"
const KEY_STATS_CANDIDATE_TRAIN = "best_candidate_train"
const KEY_STATS_CANDIDATE_TRAIN_MODIFIED = "best_candidate_train_mod"
const KEY_STATS_CANDIDATE_VALID = "best_candidate_valid"
const KEY_STATS_CANDIDATE_VALID_MODIFIED = "best_candidate_valid_mod"
const KEY_STATS_CANDIDATE_TEST = "candidate_test"
const KEY_STATS_CANDIDATE_TEST_MODIFIED = "candidate_test_mod"
const KEY_STATS_TERMINATION_REASON = "termination_reason"

function execute_index_optimization_config(config::AbstractDict)
    datasets = process_datasets_config(config[KEY_INDEX_OPTIM_DATASET])
    index = process_index_config(config[KEY_INDEX_OPTIM_INDEX])
    descriptor = process_fitness_descriptor_config(config[KEY_INDEX_OPTIM_DESCRIPTOR])
    stats_path, images_path, tensorboard_path, tensorboard_mode, step_offset = process_results_config(something(get(config, KEY_INDEX_OPTIM_RESULTS, nothing), DEFAULT_VALUE_EMPTY_CONFIG))
    
    if haskey(config, KEY_INDEX_OPTIM_PARAMETER_MODIFICATIONS)
        index = ParametersModifiedOptimizedIndex(process_parameter_modifications_config(config[KEY_INDEX_OPTIM_PARAMETER_MODIFICATIONS]), index)
    end

    logger = isempty(tensorboard_path) ? nothing : TBLogger(tensorboard_path, MAP_TENSORBOARD_MODE[tensorboard_mode], prefix = KEY_TENSORBOARD_PREFIX)

    datasets_converted = (
        dataset_train = convert_dataset_for_index_optimization_objective(datasets.dataset_train),
        dataset_valid = isnothing(datasets.dataset_valid) ? nothing : convert_dataset_for_index_optimization_objective(datasets.dataset_valid),
        dataset_test = isnothing(datasets.dataset_test) ? nothing : convert_dataset_for_index_optimization_objective(datasets.dataset_test)
    )

    is_valid_available = !isnothing(datasets_converted.dataset_valid)
    is_test_available = !isnothing(datasets_converted.dataset_test)

    train_names, train_images, train_masks, train_weights = first(datasets_converted)
    valid_names, valid_images, valid_masks, valid_weights = something(datasets_converted.dataset_valid, repeat([nothing], 4))
    test_names, test_images, test_masks, test_weights = something(datasets_converted.dataset_test, repeat([nothing], 4))

    parameters_count = get_parameters_count(index, first(train_images))

    x0, search_range, optimizer = process_optimizer_config(config[KEY_INDEX_OPTIM_OPTIMIZER], parameters_count)

    log_config(logger, config)

    train_candidate_ref = Ref(zeros(eltype(x0), parameters_count))
    train_fitness_ref = Ref(floatmax())

    log_train_progress = function (fitness, candidate, x = 0, force_candidate_log = false, step_offset = step_offset, logger = logger, train_fitness_ref = train_fitness_ref, train_candidate_ref = train_candidate_ref)
        x += step_offset
        
        log_optimizer_progress(logger, Dict("$(TAG_TRAIN)/$(NAME_METRIC_FITNESS)" => fitness), x)

        if fitness < train_fitness_ref[] || force_candidate_log
            # this could impact performance quite a lot
            log_candidate(logger, TAG_TRAIN, fitness, candidate, x)
        end

        if fitness < train_fitness_ref[]
            train_fitness_ref[] = fitness
            train_candidate_ref[] .= candidate
        end
    end

    valid_candidate_ref = Ref(zeros(eltype(x0), parameters_count))
    valid_fitness_ref = Ref(floatmax())

    log_valid_progress = function (fitness, candidate, x = 0, force_candidate_log = false, step_offset = step_offset, logger = logger, valid_fitness_ref = valid_fitness_ref, valid_candidate_ref = valid_candidate_ref)
        x += step_offset
        
        log_optimizer_progress(logger, Dict("$(TAG_VALID)/$(NAME_METRIC_FITNESS)" => fitness), x)
        
        if fitness < valid_fitness_ref[] || force_candidate_log
            log_candidate(logger, TAG_VALID, fitness, candidate, x)
        end

        if fitness < valid_fitness_ref[]
            valid_fitness_ref[] = fitness
            valid_candidate_ref[] .= candidate
        end
    end

    train_fitness, train_candidate, termination_reason = optimize_index(
        optimizer,
        index,
        descriptor,
        train_images,
        train_masks,
        train_weights,
        x0 = x0,
        search_range = search_range,
        valid_images = valid_images,
        valid_masks = valid_masks,
        valid_weights = valid_weights,
        log_train_progress = log_train_progress,
        log_valid_progress = log_valid_progress
    )

    valid_fitness = valid_fitness_ref[]
    valid_candidate = valid_candidate_ref[]

    stats = OrderedDict(
        KEY_STATS_PARAMETERS => parameters_count,
        KEY_STATS_TERMINATION_REASON => termination_reason,
        KEY_STATS_FITNESS_TRAIN => train_fitness,
        KEY_STATS_CANDIDATE_TRAIN => train_candidate
    )

    if index isa AbstractParametersModifiedOptimizedIndex
        stats[KEY_STATS_CANDIDATE_TRAIN_MODIFIED] = modify_parameters(index, train_candidate)
    end

    log_index_images(images_path, TAG_TRAIN, logger, index, train_images, train_names, train_candidate)

    if is_valid_available
        stats[KEY_STATS_FITNESS_VALID] = valid_fitness
        stats[KEY_STATS_CANDIDATE_VALID] = valid_candidate

        if index isa AbstractParametersModifiedOptimizedIndex
            stats[KEY_STATS_CANDIDATE_VALID_MODIFIED] = modify_parameters(index, valid_candidate)
        end

        log_index_images(images_path, TAG_VALID, logger, index, valid_images, valid_names, valid_candidate)
    end

    if is_test_available
        test_objective = PreAllocatedOptimizationObjective(IndexOptimizationObjective(index, test_images, test_masks, test_weights, descriptor))
        test_candidate = is_valid_available ? valid_candidate : train_candidate 
        test_fitness = test_objective(test_candidate)

        stats[KEY_STATS_FITNESS_TEST] = test_fitness
        stats[KEY_STATS_CANDIDATE_TEST] = test_candidate

        if index isa AbstractParametersModifiedOptimizedIndex
            stats[KEY_STATS_CANDIDATE_TEST_MODIFIED] = modify_parameters(index, test_candidate)
        end
        
        log_index_images(images_path, TAG_TEST, logger, index, test_images, test_names, test_candidate)
    end

    log_stats(logger, stats)

    if !isempty(stats_path)
        YAML.write_file(stats_path, stats)
    end
end

function execute_index_optimization_config(config_path::AbstractString)
    execute_index_optimization_config(YAML.load_file(config_path, dicttype = Dict{String, Any}))
end

const KEY_INDEX_EVAL_INDEX = "index"
const KEY_INDEX_EVAL_PARAMETERS = "parameters"
const KEY_INDEX_EVAL_DESCRIPTOR = "descriptor"
const KEY_INDEX_EVAL_RESULTS = "results"
const KEY_INDEX_EVAL_DATASET = "dataset"

function execute_index_evaluation_config(config::AbstractDict)
    dataset_config = config[KEY_INDEX_EVAL_DATASET]

    if length(dataset_config) > 1
        datasets = Dict([(n, convert_dataset_for_index_optimization_objective(process_dataset_config(c))) for (n, c) in dataset_config])
    else
        datasets = Dict("eval" => convert_dataset_for_index_optimization_objective(process_dataset_config(first(values(dataset_config)))))
    end

    index = process_index_config(config[KEY_INDEX_EVAL_INDEX])
    parameters = config[KEY_INDEX_EVAL_PARAMETERS]
    descriptor = process_fitness_descriptor_config(config[KEY_INDEX_EVAL_DESCRIPTOR])
    
    results = Dict{String, Float64}()

    for (name, data) in datasets
        objective = PreAllocatedOptimizationObjective(IndexOptimizationObjective(index, data.images, data.masks, data.weights, descriptor))
        
        results[name] = objective(parameters)
    end

    if haskey(config, KEY_INDEX_EVAL_RESULTS)
        YAML.write_file(config[KEY_INDEX_EVAL_RESULTS], results)
    end

    display(results)
end

function execute_index_evaluation_config(config_path::AbstractString)
    execute_index_evaluation_config(YAML.load_file(config_path, dicttype = Dict{String, Any}))
end