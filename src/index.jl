export
    AbstractOptimized,
    AbstractOptimizedTransform,
    AbstractOptimizedReduction,
    AbstractOptimizedIndex,
    AbstractOptimizedIndexCascade,
    AbstractParameterModification,
    AbstractParametersModifiedOptimizedIndex,
    AbstractIndexFitnessDescriptor

export 
    LinearTransform,
    PolynomialTransform,
    PolynomialTransformAbsBase,
    PolynomialTransformWholeExp,
    ExponentialTransform,
    ExponentialTransformAbsBase,
    ExponentialTransformWholeExp,
    BiasTransform,
    GaussianRadialBasisKernelTransform,
    ChainTransform,
    IdentityTransform,
    NormalizationTransform,
    OpReduction,
    SummationReduction,
    ProductReduction,
    DivisionReduction,
    NormalizedReduction,
    OptimizedIndex,
    OptimizedIndexCascade,
    CleanOptimizedIndex,
    IdentityParameters,
    ChainParameterModification,
    RoundedParameters,
    MinSumDropoutParameters,
    ParametersModifiedOptimizedIndex,
    TargetVisibilityIndexFitnessDescriptor

export
    create_workspace,
    get_parameters_count,
    compute_transform,
    compute_transform!,
    compute_reduction,
    compute_reduction!,
    compute_index!,
    compute_index,
    modify_parameters!,
    modify_parameters,
    compute_fitness

abstract type AbstractWorkspaced end

@mustimplement create_workspace(workspaced::AbstractWorkspaced, inputs...)

abstract type AbstractOptimized <: AbstractWorkspaced end

@mustimplement get_parameters_count(optimized::AbstractOptimized, inputs...)

abstract type AbstractOptimizedTransform <: AbstractOptimized end

@mustimplement compute_transform!(transform::AbstractOptimizedTransform, workspace, image::AbstractArray{<:Real}, parameters::AbstractArray{<:Real}, transformed::AbstractArray{<:Real})

compute_transform!(transform::AbstractOptimizedTransform, image::AbstractArray{<:Real}, parameters::AbstractArray{<:Real}, transformed::AbstractArray{<:Real}) = compute_transform!(transform, create_workspace(transform, image), image, parameters, transformed)
compute_transform(transform::AbstractOptimizedTransform, image::AbstractArray{<:Real}, parameters::AbstractArray{<:Real}) = compute_transform!(transform, image, parameters, similar(image))
get_parameters_count(::AbstractOptimizedTransform, ::AbstractArray{<:Real}) = 1
create_workspace(::AbstractOptimizedTransform, ::AbstractArray{<:Real}) = nothing

struct GaussianRadialBasisKernelTransform <: AbstractOptimizedTransform end

function compute_transform!(::GaussianRadialBasisKernelTransform, ::Any, image::AbstractArray{<:Real}, parameters::AbstractArray{<:Real}, transformed::AbstractArray{<:Real})
    transformed .= exp.(.-((image .* first(parameters)) .^ 2))
end

struct LinearTransform <: AbstractOptimizedTransform end

function compute_transform!(::LinearTransform, ::Any, image::AbstractArray{<:Real}, parameters::AbstractArray{<:Real}, transformed::AbstractArray{<:Real})
    transformed .= image .* first(parameters)
end

struct PolynomialTransform <: AbstractOptimizedTransform end

get_parameters_count(::PolynomialTransform, ::AbstractArray{<:Real}) = 3

function compute_transform!(::PolynomialTransform, ::Any, image::AbstractArray{<:Real}, parameters::AbstractArray{<:Real}, transformed::AbstractArray{<:Real})
    transformed .= ((image .* parameters[1]) .+ parameters[2]) .^ parameters[3]
end

struct PolynomialTransformAbsBase <: AbstractOptimizedTransform end

get_parameters_count(::PolynomialTransformAbsBase, ::AbstractArray{<:Real}) = 3

function compute_transform!(::PolynomialTransformAbsBase, ::Any, image::AbstractArray{<:Real}, parameters::AbstractArray{<:Real}, transformed::AbstractArray{<:Real})
    transformed .= abs.((image .* parameters[1]) .+ parameters[2]) .^ parameters[3]
end

struct PolynomialTransformWholeExp <: AbstractOptimizedTransform end

get_parameters_count(::PolynomialTransformWholeExp, ::AbstractArray{<:Real}) = 3

function compute_transform!(::PolynomialTransformWholeExp, ::Any, image::AbstractArray{<:Real}, parameters::AbstractArray{<:Real}, transformed::AbstractArray{<:Real})
    transformed .= ((image .* parameters[1]) .+ parameters[2]) .^ round(parameters[3])
end

struct ExponentialTransform <: AbstractOptimizedTransform end

function compute_transform!(::ExponentialTransform, ::Any, image::AbstractArray{<:Real}, parameters::AbstractArray{<:Real}, transformed::AbstractArray{<:Real})
    transformed .= image .^ parameters[1]
end

struct ExponentialTransformAbsBase <: AbstractOptimizedTransform end

function compute_transform!(::ExponentialTransformAbsBase, ::Any, image::AbstractArray{<:Real}, parameters::AbstractArray{<:Real}, transformed::AbstractArray{<:Real})
    transformed .= abs.(image) .^ parameters[1]
end

struct ExponentialTransformWholeExp <: AbstractOptimizedTransform end

function compute_transform!(::ExponentialTransformWholeExp, ::Any, image::AbstractArray{<:Real}, parameters::AbstractArray{<:Real}, transformed::AbstractArray{<:Real})
    transformed .= image .^ round(parameters[1])
end

struct BiasTransform <: AbstractOptimizedTransform end

function compute_transform!(::BiasTransform, ::Any, image::AbstractArray{<:Real}, parameters::AbstractArray{<:Real}, transformed::AbstractArray{<:Real})
    transformed .= image .+ first(parameters)
end

struct ChainTransform{TTransforms <: Tuple{Vararg{<:AbstractOptimizedTransform}}} <: AbstractOptimizedTransform
    transforms::TTransforms
end

ChainTransform(transforms::AbstractOptimizedTransform...) = ChainTransform(transforms)
ChainTransform(transforms::AbstractVector) = ChainTransform(Tuple(transforms))

get_parameters_count(transform::ChainTransform, image::AbstractArray{<:Real}) = sum(t -> get_parameters_count(t, image), transform.transforms, init = 0)

function create_workspace(transform::ChainTransform, image::AbstractArray{<:Real})
    return [create_workspace(t, image) for t in transform.transforms]
end

function compute_transform!(transform::ChainTransform, workspace, image::AbstractArray{<:Real}, parameters::AbstractArray{<:Real}, transformed::AbstractArray{<:Real})
    transformed .= image

    parameters_indices = eachindex(parameters)
    parameters_indices_start = 1
    
    for (t, w) in zip(transform.transforms, workspace)
        parameters_count_transform = get_parameters_count(t, transformed)

        parameters_indices_end = parameters_indices_start + parameters_count_transform - 1
        
        parameters_indices_range_start = get(parameters_indices, parameters_indices_start, last(parameters_indices))
        parameters_indices_range_end = get(parameters_indices, parameters_indices_end, first(parameters_indices) - 1)
    
        parameters_view = view(parameters, parameters_indices_range_start : parameters_indices_range_end)

        compute_transform!(t, w, transformed, parameters_view, transformed)
        
        parameters_indices_start += parameters_count_transform
    end

    return transformed
end

struct IdentityTransform <: AbstractOptimizedTransform end

get_parameters_count(::IdentityTransform, ::AbstractArray{<:Real}) = 0
compute_transform!(::IdentityTransform, ::Any, image::AbstractArray{<:Real}, ::AbstractArray{<:Real}, transformed::AbstractArray{<:Real}) = transformed .= image

struct NormalizationTransform{TNormalization} <: AbstractOptimizedTransform 
    normalization::TNormalization
end

NormalizationTransform() = NormalizationTransform(normalize_unit!)
MeanNormalizationTransform() = NormalizationTransform(normalize_mean!)
ZNormalizationTransform() = NormalizationTransform(normalize_z!)

get_parameters_count(::NormalizationTransform, ::AbstractArray{<:Real}) = 0
compute_transform!(transform::NormalizationTransform, ::Any, image::AbstractArray{<:Real}, ::AbstractArray{<:Real}, transformed::AbstractArray{<:Real}) = transformed .= transform.normalization(image)

abstract type AbstractOptimizedReduction <: AbstractOptimized end

@mustimplement compute_reduction!(reduction::AbstractOptimizedReduction, workspace, images::AbstractArray{<:AbstractArray{<:Real}}, parameters::AbstractArray{<:Real}, index::AbstractArray{<:Real})

compute_reduction!(reduction::AbstractOptimizedReduction, images::AbstractArray{<:AbstractArray{<:Real}}, parameters::AbstractArray{<:Real}, index::AbstractArray{<:Real}) = compute_reduction!(reduction, create_workspace(reduction, images), images, parameters, index)
compute_reduction(reduction::AbstractOptimizedReduction, images::AbstractArray{<:AbstractArray{<:Real}}, parameters::AbstractArray{<:Real}) = compute_reduction!(reduction, images, parameters, similar(first(images)))
get_parameters_count(::AbstractOptimizedReduction, ::AbstractArray{<:AbstractArray{<:Real}}) = 0
create_workspace(::AbstractOptimizedReduction, ::AbstractArray{<:AbstractArray{<:Real}}) = nothing

struct OpReduction{T} <: AbstractOptimizedReduction where T
    op::T
end

SummationReduction() = OpReduction(+)
ProductReduction() = OpReduction(*)
DivisionReduction() = OpReduction(/)

function compute_reduction!(reduction::OpReduction, ::Any, images::AbstractArray{<:AbstractArray{<:Real}}, parameters::AbstractArray{<:Real}, index::AbstractArray{<:Real})
    for (i, image) in enumerate(images)
        if i == 1
            index .= image
        else
            index .= reduction.op.(index, image)
        end
    end
    
    return index
end

struct NormalizedReduction{TReduction <: AbstractOptimizedReduction, TNormalization} <: AbstractOptimizedReduction
    reduction::TReduction
    normalization::TNormalization
end

NormalizedReduction(reduction::AbstractOptimizedReduction) = NormalizedReduction(reduction, normalize_unit!)

get_parameters_count(reduction::NormalizedReduction, images::AbstractArray{<:AbstractArray{<:Real}}) = get_parameters_count(reduction.reduction, images)
create_workspace(reduction::NormalizedReduction, images::AbstractArray{<:AbstractArray{<:Real}}) = create_workspace(reduction.reduction, images)

function compute_reduction!(reduction::NormalizedReduction, workspace, images::AbstractArray{<:AbstractArray{<:Real}}, parameters::AbstractArray{<:Real}, index::AbstractArray{<:Real})
    compute_reduction!(reduction.reduction, workspace, images, parameters, index)
    index .= reduction.normalization(index)
end

abstract type AbstractOptimizedIndex <: AbstractOptimized end

@mustimplement get_parameters_count(index::AbstractOptimizedIndex, images::AbstractArray{<:AbstractArray{<:Real}})
@mustimplement create_workspace(index::AbstractOptimizedIndex, images::AbstractArray{<:AbstractArray{<:Real}})
@mustimplement compute_index!(index::AbstractOptimizedIndex, workspace, images::AbstractArray{<:AbstractArray{<:Real}}, parameters::AbstractArray{<:Real}, index_image::AbstractArray{<:Real})

compute_index!(index::AbstractOptimizedIndex, images::AbstractArray{<:AbstractArray{<:Real}}, parameters::AbstractArray{<:Real}, index_image::AbstractArray{<:Real}) = compute_index!(index, create_workspace(index, images), images, parameters, index_image)
compute_index(index::AbstractOptimizedIndex, images::AbstractArray{<:AbstractArray{<:Real}}, parameters::AbstractArray{<:Real}) = compute_index!(index, images, parameters, similar(first(images)))

struct OptimizedIndex{TTransform <: AbstractOptimizedTransform, TReduction <: AbstractOptimizedReduction} <: AbstractOptimizedIndex 
    transform::TTransform
    reduction::TReduction
end

OptimizedIndex(reduction::AbstractOptimizedReduction) = OptimizedIndex(IdentityTransform(), reduction)

function create_workspace(index::OptimizedIndex, images::AbstractArray{<:AbstractArray{<:Real}})
    return (
        transformed_images = [similar(i) for i in images],
        transform_workspace = create_workspace(index.transform, first(images)),
        reduction_workspace = create_workspace(index.reduction, images)
    )
end

get_parameters_count(index::OptimizedIndex, images::AbstractArray{<:AbstractArray{<:Real}}) = get_parameters_count(index.transform, first(images)) * length(images) + get_parameters_count(index.reduction, images)

function compute_index!(index::OptimizedIndex, workspace, images::AbstractArray{<:AbstractArray{<:Real}}, parameters::AbstractArray{<:Real}, index_image::AbstractArray{<:Real})
    transformed_images = workspace.transformed_images
    
    parameters_count_transform = get_parameters_count(index.transform, first(transformed_images))
    parameters_count_reduction = get_parameters_count(index.reduction, transformed_images)

    parameters_indices = eachindex(parameters)

    parameters_indices_start = 1

    for (image, transformed_image) in zip(images, transformed_images)
        parameters_indices_end = parameters_indices_start + parameters_count_transform - 1
        
        parameters_indices_transform_range_start = get(parameters_indices, parameters_indices_start, last(parameters_indices))
        parameters_indices_transform_range_end = get(parameters_indices, parameters_indices_end, first(parameters_indices) - 1)
    
        parameters_transform_view = view(parameters, parameters_indices_transform_range_start : parameters_indices_transform_range_end)  

        compute_transform!(index.transform, workspace.transform_workspace, image, parameters_transform_view, transformed_image)

        parameters_indices_start += parameters_count_transform
    end

    parameters_indices_reduction_range_start = get(parameters_indices, parameters_indices_start, last(parameters_indices))
    parameters_indices_reduction_range_end = get(parameters_indices, parameters_indices_start + parameters_count_reduction, first(parameters_indices) - 1)

    parameters_reduction_view = view(parameters, parameters_indices_reduction_range_start : parameters_indices_reduction_range_end)

    compute_reduction!(index.reduction, workspace.reduction_workspace, transformed_images, parameters_reduction_view, index_image)
end

abstract type AbstractOptimizedIndexCascade{N} <: AbstractOptimizedIndex end

struct OptimizedIndexCascade{N, TTransform <: AbstractOptimizedTransform, TReduction <: AbstractOptimizedReduction} <: AbstractOptimizedIndexCascade{N} 
    indices::NTuple{N, <:AbstractOptimizedIndex}
    transform::TTransform
    reduction::TReduction
end

OptimizedIndexCascade(indices, reduction::AbstractOptimizedReduction) = OptimizedIndexCascade(indices, IdentityTransform(), reduction)
OptimizedIndexCascade(indices::AbstractVector{<:AbstractOptimizedIndex}, transform::AbstractOptimizedTransform, reduction::AbstractOptimizedReduction) = OptimizedIndexCascade(Tuple(indices), transform, reduction)

get_parameters_count(index::OptimizedIndexCascade{N}, images::AbstractArray{<:AbstractArray{<:Real}}) where N = sum((i, ims = images) -> get_parameters_count(i, ims), index.indices, init = 0) + N * get_parameters_count(index.transform, first(images)) + get_parameters_count(index.reduction, images)

function create_workspace(index::OptimizedIndexCascade{N}, images::AbstractArray{<:AbstractArray{<:Real}}) where N
    return (
        indices_images = [similar(first(images)) for _ in 1:N],
        indices_workspaces = [create_workspace(i, images) for i in index.indices],
        transform_workspace = create_workspace(index.transform, first(images)),
        reduction_workspace = create_workspace(index.reduction, [first(images) for _ in 1:N])
    )
end

function compute_index!(index::OptimizedIndexCascade, workspace, images::AbstractArray{<:AbstractArray{<:Real}}, parameters::AbstractArray{<:Real}, index_image::AbstractArray{<:Real})
    indices_images = workspace.indices_images
    indices_workspaces = workspace.indices_workspaces
    transform_workspace = workspace.transform_workspace
    reduction_workspace = workspace.reduction_workspace

    parameters_count_transform = get_parameters_count(index.transform, first(images))
    
    parameters_indices = eachindex(parameters)
    
    parameters_indices_start = 1
    
    for (idx, idx_w, _index_image) in zip(index.indices, indices_workspaces, indices_images)
        parameters_count_idx = get_parameters_count(idx, images)

        parameters_indices_end = parameters_indices_start + parameters_count_idx - 1
        parameters_indices_range_start = get(parameters_indices, parameters_indices_start, last(parameters_indices))
        parameters_indices_range_end = get(parameters_indices, parameters_indices_end, first(parameters_indices) - 1)

        parameters_view = view(parameters, parameters_indices_range_start : parameters_indices_range_end)

        compute_index!(idx, idx_w, images, parameters_view, _index_image)
    
        parameters_indices_start += parameters_count_idx

        parameters_indices_end = parameters_indices_start + parameters_count_transform - 1
        parameters_indices_range_start = get(parameters_indices, parameters_indices_start, last(parameters_indices))
        parameters_indices_range_end = get(parameters_indices, parameters_indices_end, first(parameters_indices) - 1)

        parameters_view = view(parameters, parameters_indices_range_start : parameters_indices_range_end)

        compute_transform!(index.transform, transform_workspace, _index_image, parameters_view, _index_image)

        parameters_indices_start += parameters_count_transform
    end

    parameters_count_reduction = get_parameters_count(index.reduction, indices_images)

    parameters_indices_end = parameters_indices_start + parameters_count_reduction - 1
    parameters_indices_range_start = get(parameters_indices, parameters_indices_start, last(parameters_indices))
    parameters_indices_range_end = get(parameters_indices, parameters_indices_end, first(parameters_indices) - 1)

    parameters_view_reduction = view(parameters, parameters_indices_range_start : parameters_indices_range_end)

    compute_reduction!(index.reduction, reduction_workspace, indices_images, parameters_view_reduction, index_image)
end

struct CleanOptimizedIndex{TIndex <: AbstractOptimizedIndex} <: AbstractOptimizedIndex
    index::TIndex
end

get_parameters_count(index::CleanOptimizedIndex, images::AbstractArray{<:AbstractArray{<:Real}}) = get_parameters_count(index.index, images)
create_workspace(index::CleanOptimizedIndex, images::AbstractArray{<:AbstractArray{<:Real}}) = create_workspace(index.index, images)

function compute_index!(index::CleanOptimizedIndex, workspace, images::AbstractArray{<:AbstractArray{<:Real}}, parameters::AbstractArray{<:Real}, index_image::AbstractArray{<:Real})
    compute_index!(index.index, workspace, images, parameters, index_image)
    replace!(index_image, Inf => maximum_finite(index_image), -Inf => minimum_finite(index_image), NaN => zero(eltype(index_image)))
end

abstract type AbstractParameterModification <: AbstractWorkspaced end

@mustimplement create_workspace(modification::AbstractParameterModification, parameters::AbstractArray{<:Real})
@mustimplement modify_parameters!(modification::AbstractParameterModification, workspace, parameters::AbstractArray{<:Real}, parameters_modified::AbstractArray{<:Real})

modify_parameters!(modification::AbstractParameterModification, parameters::AbstractArray{<:Real}, parameters_modified::AbstractArray{<:Real}) = modify_parameters!(modification, create_workspace(modification, parameters), parameters, parameters_modified)
modify_parameters(modification::AbstractParameterModification, parameters::AbstractArray{<:Real}) = modify_parameters!(modification, parameters, similar(parameters))

struct IdentityParameters <: AbstractParameterModification end

create_workspace(::IdentityParameters, ::AbstractArray{<:Real}) = nothing
modify_parameters!(::IdentityParameters, ::Nothing, parameters::AbstractArray{<:Real}, parameters_modified::AbstractArray{<:Real}) = parameters_modified .= parameters

struct ChainParameterModification{TModifications <: Tuple{Vararg{<:AbstractParameterModification}}} <: AbstractParameterModification
    modifications::TModifications
end

ChainParameterModification(modifications::AbstractParameterModification...) = ChainParameterModification(modifications)
ChainParameterModification(modifications::AbstractVector) = ChainParameterModification(Tuple(modifications))

create_workspace(modification::ChainParameterModification, parameters::AbstractArray{<:Real}) = [create_workspace(m, parameters) for m in modification.modifications]

function modify_parameters!(modification::ChainParameterModification, workspace, parameters::AbstractArray{<:Real}, parameters_modified::AbstractArray{<:Real})
    parameters_modified .= parameters

    for (w, m) in zip(workspace, modification.modifications)
        modify_parameters!(m, w, parameters_modified, parameters_modified)
    end

    return parameters_modified
end

struct RoundedParameters <: AbstractParameterModification
    digits::Int
end

create_workspace(::RoundedParameters, ::AbstractArray{<:Real}) = nothing
modify_parameters!(modification::RoundedParameters, ::Nothing, parameters::AbstractArray{<:Real}, parameters_modified::AbstractArray{<:Real}) = parameters_modified .= round.(parameters, digits = modification.digits)

struct MinSumDropoutParameters <: AbstractParameterModification
    parameter_indices::Vector{Vector{Int}}
    dropout_count::Int
end

create_workspace(modification::MinSumDropoutParameters, parameters::AbstractArray{<:Real}) = (
    parameters_sum = zeros(eltype(parameters), length(modification.parameter_indices)),
    parameters_sum_sortperm = zeros(Int, length(modification.parameter_indices))
)

function modify_parameters!(modification::MinSumDropoutParameters, workspace, parameters::AbstractArray{<:Real}, parameters_modified::AbstractArray{<:Real})
    parameters_sum = workspace.parameters_sum
    parameters_sum_sortperm = workspace.parameters_sum_sortperm

    parameters_modified .= parameters
    parameters_sum .= zero(eltype(parameters_sum))

    for (i, indices) in enumerate(modification.parameter_indices)
        for j in indices
            parameters_sum[i] += abs(parameters_modified[j]) 
        end
    end

    sortperm!(parameters_sum_sortperm, parameters_sum)

    for k in 1:modification.dropout_count
        for l in modification.parameter_indices[parameters_sum_sortperm[k]]
            parameters_modified[l] = zero(eltype(parameters_modified))
        end
    end

    return parameters_modified
end

abstract type AbstractParametersModifiedOptimizedIndex <: AbstractOptimizedIndex end

@mustimplement modify_parameters!(index::AbstractParametersModifiedOptimizedIndex, parameters::AbstractArray{<:Real}, parameters_modified::AbstractArray{<:Real})
modify_parameters(index::AbstractParametersModifiedOptimizedIndex, parameters::AbstractArray{<:Real}) = modify_parameters!(index, parameters, similar(parameters))

struct ParametersModifiedOptimizedIndex{TModification <: AbstractParameterModification, TIndex <: AbstractOptimizedIndex} <: AbstractParametersModifiedOptimizedIndex 
    modification::TModification
    index::TIndex
end

modify_parameters!(index::ParametersModifiedOptimizedIndex, workspace, parameters::AbstractArray{<:Real}, parameters_modified::AbstractArray{<:Real}) = modify_parameters!(index.modification, workspace, parameters, parameters_modified)
modify_parameters!(index::ParametersModifiedOptimizedIndex, parameters::AbstractArray{<:Real}, parameters_modified::AbstractArray{<:Real}) = modify_parameters!(index, create_workspace(index.modification, parameters), parameters, parameters_modified)
get_parameters_count(index::ParametersModifiedOptimizedIndex, images::AbstractArray{<:AbstractArray{<:Real}}) = get_parameters_count(index.index, images)
create_workspace(index::ParametersModifiedOptimizedIndex, images::AbstractArray{<:AbstractArray{<:Real}}) = (
    index_workspace = create_workspace(index.index, images),
    modification_workspace = create_workspace(index.modification, zeros(eltype(first(images)), get_parameters_count(index, images))),
    parameters_modified = zeros(eltype(first(images)), get_parameters_count(index, images))
)

function compute_index!(index::ParametersModifiedOptimizedIndex, workspace, images::AbstractArray{<:AbstractArray{<:Real}}, parameters::AbstractArray{<:Real}, index_image::AbstractArray{<:Real})
    parameters_modified = workspace.parameters_modified
    modification_workspace = workspace.modification_workspace
    index_workspace = workspace.index_workspace

    modify_parameters!(index.modification, modification_workspace, parameters, parameters_modified)

    compute_index!(index.index, index_workspace, images, parameters_modified, index_image)
end

abstract type AbstractIndexFitnessDescriptor <: AbstractWorkspaced end

@mustimplement compute_fitness(descriptor::AbstractIndexFitnessDescriptor, workspace, index_image::AbstractArray{<:Real}, target_mask::AbstractArray{<:Bool})
compute_fitness(descriptor::AbstractIndexFitnessDescriptor, index_image::AbstractArray{<:Real}, target_mask::AbstractArray{<:Bool}) = compute_fitness(descriptor, create_workspace(descriptor, index_image, target_mask), index_image, target_mask)

struct TargetVisibilityIndexFitnessDescriptor <: AbstractIndexFitnessDescriptor end

create_workspace(::TargetVisibilityIndexFitnessDescriptor, index_image::AbstractArray{<:Real}, target_mask::AbstractArray{<:Bool}) = similar(index_image)

compute_fitness(::TargetVisibilityIndexFitnessDescriptor, workspace, index_image::AbstractArray{<:Real}, target_mask::AbstractArray{<:Bool}) = -tvi(normalize_unit!(workspace, index_image), target_mask, false) 
