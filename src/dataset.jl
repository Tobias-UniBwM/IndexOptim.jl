using Images, Interpolations, StatsBase

import JSON, YAML
import Base: length, iterate, @kwdef

export 
    AbstractDataset,
    AbstractCapture,
    AbstractAnnotation,
    Capture,
    Annotation,
    AbstractAnnotationFilter,
    AnnotationCategoryFilter,
    AnnotationAttributeFilter,
    InvertedAnnotationFilter,
    AnnotationFilteredDataset,
    AbstractChannelFilter,
    ChannelNameFilter,
    InvertedChannelFilter,
    ChannelFilteredDataset,
    AbstractCaptureFilter,
    CaptureNameFilter,
    CaptureFilteredDataset,
    InvertedCaptureFilter,
    AbstractChannelExtension,
    NormalizedDifferenceChannel,
    TGIChannel,
    VARIChannel,
    SIPI2Channel,
    SAVIChannel,
    ChannelExtendedDataset,
    AbstractAttributeExtension,
    CategoryAttribute,
    CaptureNamePresenceAttribute,
    SquareMetersAttribute,
    AttributeExtendedDataset,
    ResizedDataset,
    ZNormalizedDataset,
    MinMaxNormalizedDataset,
    EqualizedDataset,
    ImageSegmentationDataset,
    AbstractFilePathIdentifier,
    DirectoryFilePathIdentifier,
    FilePathIdentifier,
    RegexFilePathIdentifier

export 
    name,
    channels,
    channel_names,
    capture_name,
    category_name,
    label_mask,
    attributes,
    capture_names,
    category_names,
    category_weights,
    capture,
    annotations,
    check_annotation,
    check_channel,
    check_capture,
    compute_channel,
    value,
    name,
    identify_file_path,
    read_capture_paths,
    read_capture_meta_paths,
    SortedCapturesCaptureIdentifier,
    SortedCapturesChannelIdentifier,
    BulkedCapturesCaptureIdentifier,
    BulkedCapturesChannelIdentifier

export 
    process_area_attribute_config,
    process_attribute_filter_config,
    process_attributes_extension_config,
    process_attributes_filter_config,
    process_capture_name_attribute_config,
    process_categories_filter_config,
    process_category_attribute_config,
    process_category_colors_config,
    process_channels_extension_config,
    process_channels_filter_config,
    process_dataset_config,
    process_dataset_extensions_config,
    process_dataset_filters_config,
    process_dataset_normalization_config,
    process_dataset_scale_config,
    process_dataset_resize_config,
    process_dataset_equalization_config,
    process_dataset_image_segmentation_config

export
    register_dataset_config_provider,
    register_dataset_config_extension,
    register_channel_extension_config_parser

abstract type AbstractDataset end
abstract type AbstractCapture end
abstract type AbstractAnnotation end

category_weights(dataset::AbstractDataset) = Dict([n => one(Float64) for n in category_names(dataset)])

@kwdef struct Capture{TChannels <: AbstractDict{<:AbstractString, <:AbstractArray{<:ImageValue}}, TAttributes <: AbstractDict{<:AbstractString}} <: AbstractCapture
    name::String
    channels::TChannels
    attributes::TAttributes = Dict{String, String}()
end

function name(capture::Capture)
    return capture.name
end

function channels(capture::Capture)
    return capture.channels
end

function channel_names(capture::Capture)
    return sort!(collect(keys(capture.channels)))
end

function attributes(capture::Capture)
    return capture.attributes
end

@kwdef struct Annotation{TAttributes <: AbstractDict{<:AbstractString}} <: AbstractAnnotation
    capture_name::String 
    category_name::String
    label_mask::BitMatrix
    attributes::TAttributes = Dict{String, String}()
end

function capture_name(annotation::Annotation)
    return annotation.capture_name
end

function category_name(annotation::Annotation)
    return annotation.category_name
end

function label_mask(annotation::Annotation)
    return annotation.label_mask
end

function attributes(annotation::Annotation)
    return annotation.attributes
end

mutable struct DatasetIterationState
    const capture_names::Vector{String}
    index::Int
end

function iterate(dataset::AbstractDataset, state::Optional{DatasetIterationState} = nothing)
    if state === nothing
        state = DatasetIterationState(capture_names(dataset), 0)
    end

    state.index += 1

    if state.index <= length(state.capture_names)
        capture_name = state.capture_names[state.index]
        iteration_tuple = ((capture = capture(dataset, capture_name), annotations = annotations(dataset, capture_name)), state)
    else
        iteration_tuple = nothing
    end

    return iteration_tuple
end

function length(dataset::AbstractDataset)
    return length(capture_names(dataset))
end

abstract type AbstractAnnotationFilter end

struct AnnotationCategoryFilter <: AbstractAnnotationFilter
    category_names::Vector{String}
end

function check_annotation(annotation_filter::AnnotationCategoryFilter, annotation::AbstractAnnotation)
    return category_name(annotation) in annotation_filter.category_names
end

struct AnnotationAttributeFilter{TValue} <: AbstractAnnotationFilter
    attribute_name::String
    attribute_values::Vector{TValue}
end

function check_annotation(annotation_filter::AnnotationAttributeFilter, annotation::AbstractAnnotation)
    annotation_attributes = attributes(annotation)
    return annotation_filter.attribute_name in keys(annotation_attributes) && annotation_attributes[annotation_filter.attribute_name] in annotation_filter.attribute_values
end

struct InvertedAnnotationFilter{TAnnotationFilter <: AbstractAnnotationFilter} <: AbstractAnnotationFilter
    annotation_filter::TAnnotationFilter
end

check_annotation(annotation_filter::InvertedAnnotationFilter, annotation::AbstractAnnotation) = !check_annotation(annotation_filter.annotation_filter, annotation)

struct AnnotationFilteredDataset{TDataset <: AbstractDataset} <: AbstractDataset
    dataset::TDataset
    annotation_filters::Vector{<:AbstractAnnotationFilter}
end

function capture_names(dataset::AnnotationFilteredDataset)
    return capture_names(dataset.dataset)
end

function category_names(dataset::AnnotationFilteredDataset)
    return category_names(dataset.dataset)
end

function capture(dataset::AnnotationFilteredDataset, capture_name::AbstractString)
    return capture(dataset.dataset, capture_name)
end

function annotations(dataset::AnnotationFilteredDataset, capture_name::AbstractString)
    return filter(a -> all(f -> check_annotation(f, a), dataset.annotation_filters), annotations(dataset.dataset, capture_name))
end

abstract type AbstractChannelFilter end

struct ChannelNameFilter <: AbstractChannelFilter
    names::Vector{String}
end

function check_channel(channel_filter::ChannelNameFilter, channel_name::AbstractString, channel_data::AbstractArray{<:ImageValue})
    return channel_name in channel_filter.names
end

struct InvertedChannelFilter{TChannelFilter <: AbstractChannelFilter} <: AbstractChannelFilter
    channel_filter::TChannelFilter
end

function check_channel(channel_filter::InvertedChannelFilter, channel_name::AbstractString, channel_data::AbstractArray{<:ImageValue})
    return !check_channel(channel_filter.channel_filter, channel_name, channel_data)
end

struct ChannelFilteredDataset{TDataset} <: AbstractDataset
    dataset::TDataset
    channel_filters::Vector{<:AbstractChannelFilter}
end

function capture_names(dataset::ChannelFilteredDataset)
    return capture_names(dataset.dataset)
end

function category_names(dataset::ChannelFilteredDataset)
    return category_names(dataset.dataset)
end

function capture(dataset::ChannelFilteredDataset, capture_name::AbstractString)
    _capture = capture(dataset.dataset, capture_name)
    _channels = channels(_capture)

    return Capture(name(_capture), filter(((n, c),) -> all(map(f -> check_channel(f, n, c), dataset.channel_filters)), _channels), attributes(_capture))
end

function annotations(dataset::ChannelFilteredDataset, capture_name::AbstractString)
    return annotations(dataset.dataset, capture_name)
end

abstract type AbstractCaptureFilter end

@kwdef struct CaptureNameFilter <: AbstractCaptureFilter
    capture_names::Vector{<:AbstractString}
    prefix::Bool = false
    suffix::Bool = false
end

CaptureNameFilter(capture_name::AbstractString, suffix::Bool, prefix::Bool) = CaptureNameFilter([capture_name], suffix, prefix)

function check_capture(capture_filter::CaptureNameFilter, capture_name::AbstractString)
    return capture_name in capture_filter.capture_names || capture_filter.prefix && any(p -> startswith(capture_name, p), capture_filter.capture_names) || capture_filter.suffix && any(s -> endswith(capture_name, s), capture_filter.capture_names)    
end

@kwdef struct InvertedCaptureFilter{TCaptureFilter <: AbstractCaptureFilter} <: AbstractCaptureFilter 
    capture_filter::TCaptureFilter
end

function check_capture(capture_filter::InvertedCaptureFilter, capture_name::AbstractString)
    return !check_capture(capture_filter.capture_filter, capture_name)
end

struct CaptureFilteredDataset{TDataset <: AbstractDataset} <: AbstractDataset
    dataset::TDataset
    capture_filters::Vector{<:AbstractCaptureFilter}
end

function capture_names(dataset::CaptureFilteredDataset)
    return filter(n -> all(f -> check_capture(f, n), dataset.capture_filters), capture_names(dataset.dataset))
end

function category_names(dataset::CaptureFilteredDataset)
    return category_names(dataset.dataset)
end

function capture(dataset::CaptureFilteredDataset, capture_name::AbstractString)
    @assert all(f -> check_capture(f, capture_name), dataset.capture_filters) "Capture with name $(capture_name) doesn't meet all filter criteria"

    return capture(dataset.dataset, capture_name)
end

function annotations(dataset::CaptureFilteredDataset, capture_name::AbstractString)
    @assert all(f -> check_capture(f, capture_name), dataset.capture_filters) "Capture with name $(capture_name) doesn't meet all filter criteria"

    return annotations(dataset.dataset, capture_name)
end

abstract type AbstractChannelExtension end

@kwdef struct NormalizedDifferenceChannel <: AbstractChannelExtension
    numerator_name::String
    denominator_name::String
    normalize_inputs::Bool = false
    normalize_output::Bool = false
end

function compute_channel(extender::NormalizedDifferenceChannel, capture::AbstractCapture)
    _channels = channels(capture)

    numerator = _channels[extender.numerator_name]
    denominator = _channels[extender.denominator_name]
    
    if extender.normalize_inputs
        numerator = normalize_unit(numerator)
        denominator = normalize_unit(denominator)
    end

    output = normalized_difference(numerator, denominator)

    if extender.normalize_output
        normalize_unit!(output)
    end

    return output
end

@kwdef struct TGIChannel <: AbstractChannelExtension
    blue_name::String = "blue"
    green_name::String = "green"
    red_name::String = "red"
    normalize_inputs::Bool = false
end

function compute_channel(extender::TGIChannel, capture::AbstractCapture)
    _channels = channels(capture)

    blue = _channels[extender.blue_name]
    green = _channels[extender.green_name]
    red = _channels[extender.red_name]

    if extender.normalize_inputs
        blue = normalize_unit(blue)
        green = normalize_unit(green)
        red = normalize_unit(red)
    end

    return normalize_unit!(tgi(blue, green, red))
end

@kwdef struct VARIChannel <: AbstractChannelExtension
    blue_name::String = "blue"
    green_name::String = "green"
    red_name::String = "red"
    normalize_inputs::Bool = false
end

function compute_channel(extender::VARIChannel, capture::AbstractCapture)
    _channels = channels(capture)
    
    blue = _channels[extender.blue_name]
    green = _channels[extender.green_name]
    red = _channels[extender.red_name]

    if extender.normalize_inputs
        blue = normalize_unit(blue)
        green = normalize_unit(green)
        red = normalize_unit(red)
    end

    return normalize_unit!(vari(blue, green, red))
end

@kwdef struct SIPI2Channel <: AbstractChannelExtension
    green_name::String = "green"
    red_name::String = "red"
    nir_name::String = "nir"
    normalize_inputs::Bool = false
end

function compute_channel(extender::SIPI2Channel, capture::AbstractCapture)
    _channels = channels(capture)

    green = _channels[extender.green_name]
    red = _channels[extender.red_name]
    nir = _channels[extender.nir_name]

    if extender.normalize_inputs
        green = normalize_unit(green)
        red = normalize_unit(red)
        nir = normalize_unit(nir)
    end

    return normalize_unit!(sipi2(green, red, nir))
end

@kwdef struct SAVIChannel <: AbstractChannelExtension
    red_name::String = "red"
    nir_name::String = "nir"
    L::Float64 = .5
    normalize_inputs::Bool = false
end

function compute_channel(extender::SAVIChannel, capture::AbstractCapture)
    _channels = channels(capture)

    red = _channels[extender.red_name]
    nir = _channels[extender.nir_name]

    if extender.normalize_inputs
        red = normalize_unit(red)
        nir = normalize_unit(nir)
    end

    return normalize_unit!(savi(red, nir, extender.L))    
end

struct ChannelExtendedDataset{TDataset} <: AbstractDataset
    dataset::TDataset
    extenders::Dict{String, <:AbstractChannelExtension}
end

function capture_names(dataset::ChannelExtendedDataset)
    return capture_names(dataset.dataset)
end

function category_names(dataset::ChannelExtendedDataset)
    return category_names(dataset.dataset)
end

function capture(dataset::ChannelExtendedDataset, capture_name::AbstractString)
    _capture = capture(dataset.dataset, capture_name)
    _channels = channels(_capture)

    _channels_extended = Dict{String, AbstractArray{<:ImageValue}}([n => compute_channel(e, _capture) for (n, e) in dataset.extenders])

    return Capture(name(_capture), merge!(_channels_extended, _channels), attributes(_capture))
end

function annotations(dataset::ChannelExtendedDataset, capture_name::AbstractString)
    return annotations(dataset.dataset, capture_name)
end

abstract type AbstractAttributeExtension end

struct CategoryAttribute <: AbstractAttributeExtension
    name::String
    category_map::Dict{String}
end

function value(attribute::CategoryAttribute, annotation::Annotation, capture::Capture)
    return attribute.category_map[category_name(annotation)]
end

function name(attribute::CategoryAttribute)
    return attribute.name
end

struct CaptureNamePresenceAttribute <: AbstractAttributeExtension
    name::String
    presence_map::Dict{String}
end

function value(attribute::CaptureNamePresenceAttribute, annotation::Annotation, capture::Capture)
    return last(first(filter(((key, value),) -> occursin(key, name(capture)), attribute.presence_map)))
end

function name(attribute::CaptureNamePresenceAttribute)
    return attribute.name
end

@kwdef struct SquareMetersAttribute{TGSD <: Real} <: AbstractAttributeExtension
    name::String = "sqm"
    gsd::TGSD
end

SquareMetersAttribute(gsd) = SquareMetersAttribute(gsd = gsd)

function value(attribute::SquareMetersAttribute, annotation::Annotation, capture::Capture)
    return sum(label_mask(annotation)) * (attribute.gsd ^ 2)
end

function name(attribute::SquareMetersAttribute)
    return attribute.name
end

struct AttributeExtendedDataset <: AbstractDataset
    dataset::AbstractDataset
    attributes::Vector{<:AbstractAttributeExtension}
end

function capture_names(dataset::AttributeExtendedDataset)
    return capture_names(dataset.dataset)
end

function category_names(dataset::AttributeExtendedDataset)
    return category_names(dataset.dataset)
end

function capture(dataset::AttributeExtendedDataset, capture_name::AbstractString)
    return capture(dataset.dataset, capture_name)
end

function attributes(dataset::AttributeExtendedDataset, annotation::AbstractAnnotation, capture::AbstractCapture)
    return Dict(map(a -> name(a) => value(a, annotation, capture), dataset.attributes))
end

function annotations(dataset::AttributeExtendedDataset, capture_name::AbstractString)
    _annotations = annotations(dataset.dataset, capture_name)
    _capture = capture(dataset, capture_name)

    return map(a -> Annotation(capture_name, category_name(a), label_mask(a), merge!(attributes(dataset, a, _capture), attributes(a))), _annotations)
end

struct ResizedDataset <: AbstractDataset
    dataset::AbstractDataset
    resolution::Tuple{Int, Int}
    ratio::Tuple{Float64, Float64}
    degree::Interpolations.Degree
    min_mask::Int
end

ResizedDataset(dataset::AbstractDataset, resolution::Tuple{<:Integer, <:Integer}, degree::Interpolations.Degree = Linear(), min_mask::Integer = 5) = ResizedDataset(dataset, resolution, (0., 0.), degree, min_mask)
ResizedDataset(dataset::AbstractDataset, ratio::Tuple{<:AbstractFloat, <:AbstractFloat}, degree::Interpolations.Degree = Linear(), min_mask::Integer = 5) = ResizedDataset(dataset, (0, 0), ratio, degree, min_mask)
ResizedDataset(dataset::AbstractDataset, ratio_resolution::AbstractVector{<:Real}, degree::Interpolations.Degree = Linear(), min_mask::Integer = 5) = ResizedDataset(dataset, Tuple(ratio_resolution), degree, min_mask)
ResizedDataset(dataset::AbstractDataset, ratio_resolution::Real, degree::Interpolations.Degree = Linear(), min_mask::Integer = 5) = ResizedDataset(dataset, (ratio_resolution, ratio_resolution), degree, min_mask)

function capture_names(dataset::ResizedDataset)
    return capture_names(dataset.dataset)
end

function category_names(dataset::ResizedDataset)
    return category_names(dataset.dataset)
end

function capture(dataset::ResizedDataset, capture_name::AbstractString)
    _capture = capture(dataset.dataset, capture_name)

    _channels = channels(_capture)

    for channel_name in collect(keys(_channels))
        channel_image = _channels[channel_name]
        resolution = size(channel_image)[1:2]

        if resolution != dataset.resolution && all(x -> x > 0, dataset.resolution)
            channel_image = imresize(channel_image, dataset.resolution)
        elseif all(x -> x > 0., dataset.ratio) && any(x -> x != 1., dataset.ratio)
            channel_image = imresize(channel_image, ratio = dataset.ratio)
        end

        _channels[channel_name] = channel_image
    end

    return Capture(name(_capture),_channels, attributes(_capture))
end

function annotations(dataset::ResizedDataset, _capture_name::AbstractString)
    resized_annotations = map(annotations(dataset.dataset, _capture_name)) do a
        _label_mask = label_mask(a)
        resolution = size(_label_mask)
        
        if resolution != dataset.resolution && all(x -> x > 0, dataset.resolution)
            _label_mask = convert.(Bool, imresize(_label_mask, dataset.resolution, method = Constant()))
        elseif all(x -> x > 0., dataset.ratio) && any(x -> x != 1., dataset.ratio)
            _label_mask = convert.(Bool, imresize(_label_mask, ratio = dataset.ratio, method = Constant()))
        end

        return Annotation(capture_name(a), category_name(a), _label_mask, attributes(a))
    end

    return filter(a -> sum(label_mask(a)) >= dataset.min_mask, resized_annotations)
end

function compute_channel_means(dataset::AbstractDataset)
    means = Dict{String, Vector{Float64}}()

    _capture_names = capture_names(dataset)
    
    # compute means
    for _capture_name in _capture_names
        _capture = capture(dataset, _capture_name)
        _capture_channels = channels(_capture)

        for (channel_name, channel_image) in _capture_channels
            channel_image_f = convert.(Float64, channelview(channel_image))
            channel_image_mean = vec(mean(channel_image_f, dims = (length(size(channel_image_f)) - 1, length(size(channel_image_f)))) ./ length(_capture_names))

            if !haskey(means, channel_name)
                means[channel_name] = channel_image_mean
            else
                means[channel_name] .+= channel_image_mean
            end
        end
    end

    return means
end

function compute_channel_stds(dataset::AbstractDataset, means::AbstractDict{<:AbstractString, <:AbstractVector{<:Real}} = compute_channel_means(dataset); corrected::Bool = true)
    stds = Dict{String, Vector{Float64}}()
    
    _capture_names = capture_names(dataset)

    for _capture_name in _capture_names
        _capture = capture(dataset, _capture_name)
        _capture_channels = channels(_capture)

        for (channel_name, channel_image) in _capture_channels
            channel_image_f = convert.(Float64, channelview(channel_image))
            channel_image_std = vec(var(channel_image_f, corrected = false, mean = means[channel_name], dims = (length(size(channel_image_f)) - 1, length(size(channel_image_f)))) ./ (length(_capture_names) - corrected))

            if !haskey(stds, channel_name)
                stds[channel_name] = channel_image_std
            else
                stds[channel_name] .+= channel_image_std
            end
        end
    end

    for channel_std in values(stds)
        channel_std .= sqrt.(channel_std)
    end

    return stds
end

struct ZNormalizedDataset <: AbstractDataset
    dataset::AbstractDataset
    mean::Dict{String, Vector{Float64}}
    std::Dict{String, Vector{Float64}}
end

function ZNormalizedDataset(dataset::AbstractDataset; corrected::Bool = true)
    means = compute_channel_means(dataset)
    stds = compute_channel_stds(dataset, means, corrected = corrected)

    return ZNormalizedDataset(dataset, means, stds)
end

function capture_names(dataset::ZNormalizedDataset)
    return capture_names(dataset.dataset)
end

function category_names(dataset::ZNormalizedDataset)
    return category_names(dataset.dataset)
end

function capture(dataset::ZNormalizedDataset, capture_name::AbstractString)
    _capture = capture(dataset.dataset, capture_name)

    _channels = Dict{String, AbstractArray{Float64}}()

    for (channel_name, channel_image) in channels(_capture)
        channel_image_float = convert.(Float64, channelview(channel_image))
        channel_image_depth = length(size(channel_image_float)) > 2 ? first(size(channel_image_float)) : 1

        channel_image_float_v = reshape(channel_image_float, channel_image_depth, :)
        
        if haskey(dataset.mean, channel_name) && haskey(dataset.std, channel_name)
            normalize_z!(channel_image_float_v, dataset.mean[channel_name], dataset.std[channel_name])
        end

        _channels[channel_name] = channel_image_float
    end

    return Capture(name(_capture), _channels, attributes(_capture))
end

function annotations(dataset::ZNormalizedDataset, capture_name::AbstractString)
    return annotations(dataset.dataset, capture_name)
end

function compute_channel_extrema(dataset::AbstractDataset)
    _extrema = Dict{String, Vector{Vector{Float64}}}()

    _capture_names = capture_names(dataset)
    
    # compute extrema
    for _capture_name in _capture_names
        _capture = capture(dataset, _capture_name)
        _capture_channels = channels(_capture)

        for (channel_name, channel_image) in _capture_channels
            channel_image_f = convert.(Float64, channelview(channel_image))

            if length(size(channel_image_f)) > 2
                channel_extrema = [collect(extrema(view(channel_image_f, i, :, :))) for i in first(axes(channel_image_f))]
            else
                channel_extrema = [collect(extrema(channel_image_f))]
            end

            if !haskey(_extrema, channel_name)
                _extrema[channel_name] = channel_extrema
            else
                existing_channel_extrema = _extrema[channel_name]

                for i in eachindex(channel_extrema)
                    if channel_extrema[i][1] < existing_channel_extrema[i][1]
                        existing_channel_extrema[i][1] = channel_extrema[i][1]    
                    elseif channel_extrema[i][2] > existing_channel_extrema[i][2]
                        existing_channel_extrema[i][2] = channel_extrema[i][2]
                    end     
                end
                
            end
        end
    end

    return _extrema
end

struct MinMaxNormalizedDataset <: AbstractDataset
    dataset::AbstractDataset
    extrema::Dict{String, Vector{Vector{Float64}}}
end

function MinMaxNormalizedDataset(dataset::AbstractDataset; corrected::Bool = true)
    extrema = compute_channel_extrema(dataset)
    
    return MinMaxNormalizedDataset(dataset, extrema)
end

function capture_names(dataset::MinMaxNormalizedDataset)
    return capture_names(dataset.dataset)
end

function category_names(dataset::MinMaxNormalizedDataset)
    return category_names(dataset.dataset)
end

function capture(dataset::MinMaxNormalizedDataset, capture_name::AbstractString)
    _capture = capture(dataset.dataset, capture_name)

    _channels = Dict{String, AbstractArray{Float64}}()

    for (channel_name, channel_image) in channels(_capture)
        channel_image_float = convert.(Float64, channelview(channel_image))
        channel_image_depth = length(size(channel_image_float)) > 2 ? first(size(channel_image_float)) : 1

        channel_image_float_v = reshape(channel_image_float, channel_image_depth, :)
        
        if haskey(dataset.extrema, channel_name)
            channel_extrema = dataset.extrema[channel_name]

            for i in first(axes(channel_image_float_v))
                channel_image_float_v[i, :] .-= channel_extrema[i][1]
                channel_image_float_v[i, :] ./= channel_extrema[i][2] - channel_extrema[i][1]
            end
        end

        _channels[channel_name] = channel_image_float
    end

    return Capture(name(_capture), _channels, attributes(_capture))
end

function annotations(dataset::MinMaxNormalizedDataset, capture_name::AbstractString)
    return annotations(dataset.dataset, capture_name)
end

struct EqualizedDataset{TDataset <: AbstractDataset} <: AbstractDataset
    dataset::TDataset
    category_weights::Dict{String, Float64}
end

function EqualizedDataset(dataset::AbstractDataset)
    annotation_histogram = Dict([n => zero(Int) for n in category_names(dataset)])

    for _capture_name in capture_names(dataset)
        _annotations = annotations(dataset, _capture_name)

        for _annotation in _annotations
            annotation_histogram[category_name(_annotation)] += 1
        end
    end

    for _category_name in collect(keys(annotation_histogram))
        if annotation_histogram[_category_name] <= 0
            pop!(annotation_histogram, _category_name)
        end
    end

    annotations_sum = sum(values(annotation_histogram))
    category_count = length(annotation_histogram)

    category_weights = Dict([n => annotations_sum / (category_count * c) for (n, c) in annotation_histogram])

    return EqualizedDataset(dataset, category_weights)
end

capture_names(dataset::EqualizedDataset) = capture_names(dataset.dataset)
category_names(dataset::EqualizedDataset) = category_names(dataset.dataset)
category_weights(dataset::EqualizedDataset) = dataset.category_weights
capture(dataset::EqualizedDataset, capture_name::AbstractString) = capture(dataset.dataset, capture_name)
annotations(dataset::EqualizedDataset, capture_name::AbstractString) = annotations(dataset.dataset, capture_name)

@kwdef struct ImageSegmentationDataset{TCategoryColor <: Colorant} <: AbstractDataset 
    capture_paths::Dict{String, Dict{String, String}}
    label_paths::Dict{String, String}
    meta_paths::Dict{String, String} = Dict{String, String}()
    category_colors::Dict{String, TCategoryColor}
    min_label_mask_size::Int = 5
end

function capture_names(dataset::ImageSegmentationDataset)
    return sort!(collect(keys(dataset.capture_paths)))
end

function category_names(dataset::ImageSegmentationDataset)
    return sort!(collect(keys(dataset.category_colors)))
end

function capture(dataset::ImageSegmentationDataset, capture_name::AbstractString)
    channel_paths = dataset.capture_paths[capture_name]
    meta_paths = dataset.meta_paths

    channels = Dict{String, Matrix{<:Colorant}}(map(n -> n => load(channel_paths[n]), collect(keys(channel_paths))))
    attributes = haskey(meta_paths, capture_name) ? JSON.parsefile(meta_paths[capture_name]) : Dict{String, Any}()

    return Capture(name = capture_name, channels = channels, attributes = attributes)
end

function annotations(dataset::ImageSegmentationDataset, capture_name::AbstractString)
    label_path = dataset.label_paths[capture_name]
    label_mask_colored = load(label_path)
    label_mask_unique_colors = unique(label_mask_colored)

    annotations = Annotation[]

    for color in label_mask_unique_colors
        if color in values(dataset.category_colors)
            category_name = first(keys(filter(((n, c), ) -> c == color, dataset.category_colors)))
            category_label_mask = label_mask_colored .== color
            category_label_mask_components = label_components(category_label_mask, trues(3, 3))
    
            for component_label in unique(category_label_mask_components)
                component_label_mask = category_label_mask_components .== component_label

                if component_label != 0 && sum(component_label_mask) >= dataset.min_label_mask_size
                    push!(annotations, Annotation(capture_name = capture_name, category_name = category_name, label_mask = category_label_mask_components .== component_label))
                end
            end
        end
    end

    return annotations
end

abstract type AbstractFilePathIdentifier end

@kwdef struct DirectoryFilePathIdentifier <: AbstractFilePathIdentifier
    root_directory::String
    path_delimiter_replacement::String = "_"
end

DirectoryFilePathIdentifier(root_directory::AbstractString) = DirectoryFilePathIdentifier(root_directory = root_directory)

function identify_file_path(identifier::DirectoryFilePathIdentifier, file_path::AbstractString)
    ident = relpath(dirname(file_path), identifier.root_directory)

    if !isempty(identifier.path_delimiter_replacement)
        ident = join(splitpath(ident), identifier.path_delimiter_replacement)
    end

    return ident
end

@kwdef struct FilePathIdentifier <: AbstractFilePathIdentifier 
    ignore_extension::Bool = true
    ignore_directory::Bool = true
end

function identify_file_path(identifier::FilePathIdentifier, file_path::AbstractString)
    file_path = identifier.ignore_extension ? first(splitext(file_path)) : file_path
    file_path = identifier.ignore_directory ? last(splitdir(file_path)) : file_path
    
    return file_path
end

@kwdef struct RegexFilePathIdentifier <: AbstractFilePathIdentifier
    pattern::Regex
    ignore_directory::Bool = true
    ignore_extension::Bool = true
    match_index::Int = 0
end

RegexFilePathIdentifier(pattern::Regex) = RegexFilePathIdentifier(pattern = pattern)

function identify_file_path(identifier::RegexFilePathIdentifier, file_path::AbstractString)
    file_path = identifier.ignore_extension ? first(splitext(file_path)) : file_path
    file_path = identifier.ignore_directory ? last(splitdir(file_path)) : file_path
    
    regex_match = match(identifier.pattern, file_path)

    return identifier.match_index > 0 ? regex_match[identifier.match_index] : regex_match.match
end

SortedCapturesCaptureIdentifier(captures_root_directory::AbstractString; path_delimiter_replacement::AbstractString = "_") = DirectoryFilePathIdentifier(captures_root_directory, path_delimiter_replacement)
SortedCapturesChannelIdentifier() = FilePathIdentifier(true, true)

BulkedCapturesCaptureIdentifier(capture_channel_name_delimiter::AbstractString = "_") = RegexFilePathIdentifier(Regex("(\\w+)($(capture_channel_name_delimiter))(\\w+)"), true, true, 1)
BulkedCapturesChannelIdentifier(capture_channel_name_delimiter::AbstractString = "_") = RegexFilePathIdentifier(Regex("(\\w+)($(capture_channel_name_delimiter))(\\w+)"), true, true, 3)

function read_capture_paths(directory::AbstractString, capture_identifier::AbstractFilePathIdentifier, channel_identifier::AbstractFilePathIdentifier; extensions = (".png", ".jpeg", ".jpg", ".tif", ".tiff"))
    file_paths = read_file_paths(directory, include_subdirs = true, extensions = extensions)
    capture_paths = Dict{String, Dict{String, String}}()

    for file_path in file_paths
        capture_ident = identify_file_path(capture_identifier, file_path)
        channel_ident = identify_file_path(channel_identifier, file_path)

        if !(capture_ident in keys(capture_paths))
            capture_paths[capture_ident] = Dict{String, String}()
        end

        capture_paths[capture_ident][channel_ident] = file_path
    end

    return capture_paths
end

function read_capture_meta_paths(directory::AbstractString, capture_identifier::AbstractFilePathIdentifier, meta_file_name::AbstractString)
    file_paths = read_file_paths(directory, include_subdirs = true, extensions = (pathext(meta_file_name), ))
    meta_paths = Dict{String, String}()

    for file_path in file_paths
        if last(splitpath(file_path)) == meta_file_name
            meta_paths[identify_file_path(capture_identifier, file_path)] = file_path
        end
    end

    return meta_paths
end

function process_category_colors_config(config::AbstractDict, colors_are_ubyte::Bool = true)
    categories = Dict{String, RGB}()
    
    for (category_name, category_color_vector) in config
        if colors_are_ubyte
            category_color = RGB(reinterpret.(N0f8, convert.(UInt8, category_color_vector))...)
        else
            category_color = RGB(convert.(N0f8, category_color_vector)...)
        end

        categories[category_name] = category_color
    end

    return categories
end

KEY_IMSEG_CAPTURES = "captures"
KEY_IMSEG_STRUCTURE = "structure"
KEY_IMSEG_LABEL = "label"
KEY_IMSEG_EXTENSIONS = "extensions"
KEY_IMSEG_CATEGORIES = "categories"
KEY_IMSEG_MIN_MASK = "minmask"
KEY_IMSEG_META = "meta"

VALUE_IMSEG_STRUCTURE_SORTED = "sorted"
VALUE_IMSEG_STRUCTURE_BULKED = "bulked"

DEFAULT_VALUE_IMSEG_EXTENSIONS = (".png", ".jpeg", ".jpg", ".tif", ".tiff")
DEFAULT_VALUE_IMSEG_MIN_MASK = 5
DEFAULT_VALUE_IMSEG_META = "meta.json"

function process_dataset_image_segmentation_config(config::AbstractDict)
    captures_directory = config[KEY_IMSEG_CAPTURES]
    captures_structure = config[KEY_IMSEG_STRUCTURE]
    label_channel_name = config[KEY_IMSEG_LABEL]
    category_colors_config = config[KEY_IMSEG_CATEGORIES]
    meta_file_name = get(config, KEY_IMSEG_META, DEFAULT_VALUE_IMSEG_META)
    min_mask = get(config, KEY_IMSEG_MIN_MASK, DEFAULT_VALUE_IMSEG_MIN_MASK)
    extensions = get(config, KEY_IMSEG_EXTENSIONS, DEFAULT_VALUE_IMSEG_EXTENSIONS)

    if captures_structure == VALUE_IMSEG_STRUCTURE_SORTED
        capture_identifier = SortedCapturesCaptureIdentifier(captures_directory)
        channel_identifier = SortedCapturesChannelIdentifier()
    else
        capture_identifier = BulkedCapturesCaptureIdentifier()
        channel_identifier = BulkedCapturesChannelIdentifier()
    end

    capture_paths = read_capture_paths(captures_directory, capture_identifier, channel_identifier, extensions = extensions)

    label_paths = Dict{String, String}()

    for (capture_ident, channel_paths) in capture_paths
        label_paths[capture_ident] = pop!(channel_paths, label_channel_name) 
    end

    meta_paths = read_capture_meta_paths(captures_directory, capture_identifier, meta_file_name)

    category_colors = process_category_colors_config(category_colors_config)

    return ImageSegmentationDataset(capture_paths, label_paths, meta_paths, category_colors, min_mask)
end

function process_capture_name_attribute_config(config::AbstractDict)
    return map(n -> CaptureNamePresenceAttribute(n, config[n]), collect(keys(config)))
end

KEY_AREA_ATTRIBUTE_NAME = "name"
KEY_AREA_ATTRIBUTE_GSD = "gsd"

function process_area_attribute_config(config::AbstractDict)
    return KEY_AREA_ATTRIBUTE_NAME in keys(config) ? 
        SquareMetersAttribute(config[KEY_AREA_ATTRIBUTE_NAME], config[KEY_AREA_ATTRIBUTE_GSD]) :
        SquareMetersAttribute(config[KEY_AREA_ATTRIBUTE_GSD])
end

function process_category_attribute_config(config::AbstractDict)
    return map(n -> CategoryAttribute(n, config[n]), collect(keys(config)))
end

KEY_AREA_ATTRIBUTE = "area"
KEY_CAPTURE_NAME_ATTRIBUTE = "name"
KEY_CATEGORY_ATTRIBUTE = "category"

function process_attributes_extension_config(config::AbstractDict)
    attributes = AbstractAttributeExtension[]

    if KEY_AREA_ATTRIBUTE_NAME in keys(config)
        push!(attributes, process_area_attribute_config(config[KEY_AREA_ATTRIBUTE]))
    end

    if KEY_CAPTURE_NAME_ATTRIBUTE in keys(config)
        append!(attributes, process_capture_name_attribute_config(config[KEY_CAPTURE_NAME_ATTRIBUTE]))
    end

    if KEY_CATEGORY_ATTRIBUTE in keys(config)
        append!(attributes, process_category_attribute_config(config[KEY_CATEGORY_ATTRIBUTE]))
    end

    return attributes
end

const MAP_CHANNEL_EXTENSION_PARSER = Dict{String, Any}()

function register_channel_extension_config_parser(name::AbstractString, parser)
    MAP_CHANNEL_EXTENSION_PARSER[name] = parser
end

KEY_CHANNEL_TYPE = "type"
KEY_CHANNEL_ARGS = "args"

DEFAULT_VALUE_CHANNEL_ARGS = Dict{String, String}()

function process_channels_extension_config(config::AbstractDict)
    channel_extensions = Dict{String, AbstractChannelExtension}()
    
    for (extension_name, extension_config) in config
        extension_type = extension_config[KEY_CHANNEL_TYPE]
        extension_args = get(extension_config, KEY_CHANNEL_ARGS, DEFAULT_VALUE_CHANNEL_ARGS)

        if haskey(MAP_CHANNEL_EXTENSION_PARSER, extension_type)
            extension = MAP_CHANNEL_EXTENSION_PARSER[extension_type](extension_args)
        else
            extension = eval(Symbol(extension_type))(; keys_as_symbols(extension_args)...)
        end

        channel_extensions[extension_name] = extension
    end

    return channel_extensions
end

KEY_EXTENSION_CHANNELS = "channels"
KEY_EXTENSION_ATTRIBUTES = "attributes"

function process_dataset_extensions_config(config::AbstractDict, dataset::AbstractDataset)
    if KEY_EXTENSION_CHANNELS in keys(config)
        dataset = ChannelExtendedDataset(dataset, process_channels_extension_config(config[KEY_EXTENSION_CHANNELS]))
    end

    if KEY_EXTENSION_ATTRIBUTES in keys(config)
        dataset = AttributeExtendedDataset(dataset, process_attributes_extension_config(config[KEY_EXTENSION_ATTRIBUTES]))
    end

    return dataset
end

KEY_CHANNELS_FILTER_NAMES = "names"
KEY_CHANNELS_FILTER_INVERT = "invert"

function process_channels_filter_config(config::AbstractDict)
    channels_filter = ChannelNameFilter(config[KEY_CHANNELS_FILTER_NAMES])

    if get(config, KEY_CHANNELS_FILTER_INVERT, false)
        channels_filter = InvertedChannelFilter(channels_filter)
    end

    return channels_filter
end

KEY_FILTER_CATEGORIES_NAMES = "names"
KEY_FILTER_CATEGORIES_INVERT = "invert"

function process_categories_filter_config(config::AbstractDict)
    annotation_filter = AnnotationCategoryFilter(config[KEY_FILTER_CATEGORIES_NAMES])

    if get(config, KEY_FILTER_CATEGORIES_INVERT, false)
        annotation_filter = InvertedAnnotationFilter(annotation_filter)
    end

    return annotation_filter
end

KEY_FILTER_ATTRIBUTES_VALUES = "values"
KEY_FILTER_ATTRIBUTES_INVERT = "invert"

function process_attribute_filter_config(config::AbstractDict, attribute_name::AbstractString)
    annotation_filter = AnnotationAttributeFilter(attribute_name, config[KEY_FILTER_ATTRIBUTES_VALUES])

    if get(config, KEY_FILTER_ATTRIBUTES_INVERT, false)
        annotation_filter = InvertedAnnotationFilter(annotation_filter)
    end

    return annotation_filter
end

function process_attributes_filter_config(config::AbstractDict)
    return map(((n, c), ) -> process_attribute_filter_config(c, n), collect(config))
end

KEY_FILTER_CAPTURE_NAMES = "names"
KEY_FILTER_CAPTURE_PREFIX = "prefix"
KEY_FILTER_CAPTURE_SUFFIX = "suffix"
KEY_FILTER_CAPTURE_INVERT = "invert"

function process_capture_filter_config(config::AbstractDict)
    capture_filter = CaptureNameFilter(config[KEY_FILTER_CAPTURE_NAMES], get(config, KEY_FILTER_CAPTURE_PREFIX, false), get(config, KEY_FILTER_CAPTURE_SUFFIX, false))

    if get(config, KEY_FILTER_CAPTURE_INVERT, false)
        capture_filter = InvertedCaptureFilter(capture_filter)
    end

    return capture_filter
end

function process_capture_filter_configs(configs)
    return map(c -> process_capture_filter_config(c), configs)
end

KEY_FILTER_CHANNELS = "channels"
KEY_FILTER_CATEGORIES = "categories"
KEY_FILTER_ATTRIBUTES = "attributes"
KEY_FILTER_CAPTURES = "captures"

function process_dataset_filters_config(config::AbstractDict, dataset::AbstractDataset)
    if KEY_FILTER_CHANNELS in keys(config)
        dataset = ChannelFilteredDataset(dataset, [process_channels_filter_config(config[KEY_FILTER_CHANNELS])])
    end

    if KEY_FILTER_CATEGORIES in keys(config)
        dataset = AnnotationFilteredDataset(dataset, [process_categories_filter_config(config[KEY_FILTER_CATEGORIES])])
    end

    if KEY_FILTER_ATTRIBUTES in keys(config)
        dataset = AnnotationFilteredDataset(dataset, process_attributes_filter_config(config[KEY_FILTER_ATTRIBUTES]))
    end

    if KEY_FILTER_CAPTURES in keys(config)
        dataset = CaptureFilteredDataset(dataset, process_capture_filter_configs(config[KEY_FILTER_CAPTURES]))
    end

    return dataset
end

const KEY_RESIZE_RATIO = "ratio"
const KEY_RESIZE_RESOLUTION = "resolution"
const KEY_RESIZE_MIN_MASK = "minmask"

const DEFAULT_VALUE_RESIZE_MIN_MASK = 5

function process_dataset_resize_config(config::AbstractDict, dataset::AbstractDataset)
    ResizedDataset(dataset, haskey(config, KEY_RESIZE_RATIO) ? config[KEY_RESIZE_RATIO] : config[KEY_RESIZE_RESOLUTION], Linear(), get(config, KEY_RESIZE_MIN_MASK, DEFAULT_VALUE_RESIZE_MIN_MASK))
end

const KEY_NORMALIZATION_MEANS = "means"
const KEY_NORMALIZATION_STDS = "stds"
const KEY_NORMALIZATION_FILE = "file"
const KEY_NORMALIZATION_SAVE = "save"
const KEY_NORMALIZATION_LOAD = "load"
const KEY_NORMALIZATION_FORCE_LOAD = "force_load"
const KEY_NORMALIZATION_FORCE_SAVE = "force_save"

function process_dataset_normalization_config(config::AbstractDict, dataset::AbstractDataset)
    if haskey(config, KEY_NORMALIZATION_MEANS) && haskey(config, KEY_NORMALIZATION_STDS)
        means, stds = config[KEY_NORMALIZATION_MEANS], config[KEY_NORMALIZATION_STDS]
    elseif get(config, KEY_NORMALIZATION_LOAD, true) && haskey(config, KEY_NORMALIZATION_FILE) && (isfile(config[KEY_NORMALIZATION_FILE]) || get(config, KEY_NORMALIZATION_FORCE_LOAD, false))
        stats_config = YAML.load_file(config[KEY_NORMALIZATION_FILE], dicttype = Dict{String, Any})
        means, stds = stats_config[KEY_NORMALIZATION_MEANS], stats_config[KEY_NORMALIZATION_STDS]
    else
        means = compute_channel_means(dataset)
        stds = compute_channel_stds(dataset, means)
    end

    if get(config, KEY_NORMALIZATION_SAVE, true) && haskey(config, KEY_NORMALIZATION_FILE) && (!isfile(config[KEY_NORMALIZATION_FILE]) || get(config, KEY_NORMALIZATION_FORCE_SAVE, false))
        YAML.write_file(config[KEY_NORMALIZATION_FILE], Dict(KEY_NORMALIZATION_MEANS => means, KEY_NORMALIZATION_STDS => stds))
    end

    return ZNormalizedDataset(dataset, means, stds)
end

const KEY_SCALE_EXTREMA = "extrema"
const KEY_SCALE_FILE = "file"
const KEY_SCALE_SAVE = "save"
const KEY_SCALE_LOAD = "load"
const KEY_SCALE_FORCE_LOAD = "force_load"
const KEY_SCALE_FORCE_SAVE = "force_save"

function process_dataset_scale_config(config::AbstractDict, dataset::AbstractDataset)
    if haskey(config, KEY_SCALE_EXTREMA)
        _extrema = config[KEY_SCALE_EXTREMA]
    elseif get(config, KEY_SCALE_LOAD, true) && haskey(config, KEY_SCALE_FILE) && (isfile(config[KEY_SCALE_FILE]) || get(config, KEY_SCALE_FORCE_LOAD, false))
        _extrema = YAML.load_file(config[KEY_SCALE_FILE], dicttype = Dict{String, Any})
    else
        _extrema = compute_channel_extrema(dataset)
    end

    if get(config, KEY_SCALE_SAVE, true) && haskey(config, KEY_SCALE_FILE) && (!isfile(config[KEY_SCALE_FILE]) || get(config, KEY_SCALE_FORCE_SAVE, false))
        YAML.write_file(config[KEY_SCALE_FILE], _extrema)
    end

    return MinMaxNormalizedDataset(dataset, _extrema)
end

process_dataset_scale_config(::Nothing, dataset::AbstractDataset) = process_dataset_scale_config(Dict(), dataset)

const KEY_EQUALIZATION_WEIGHTS = "weights"
const KEY_EQUALIZATION_FILE = "file"
const KEY_EQUALIZATION_SAVE = "save"
const KEY_EQUALIZATION_LOAD = "load"
const KEY_EQUALIZATION_FORCE_LOAD = "force_load"
const KEY_EQUALIZATION_FORCE_SAVE = "force_save"

function process_dataset_equalization_config(config::AbstractDict, dataset::AbstractDataset)
    if haskey(config, KEY_EQUALIZATION_WEIGHTS)
        dataset = EqualizedDataset(dataset, convert(Dict{String, Float64}, config[KEY_EQUALIZATION_WEIGHTS]))
    elseif get(config, KEY_EQUALIZATION_LOAD, true) && haskey(config, KEY_EQUALIZATION_FILE) && (isfile(config[KEY_EQUALIZATION_FILE]) || get(config, KEY_EQUALIZATION_FORCE_LOAD, false))
        dataset = EqualizedDataset(dataset, YAML.load_file(config[KEY_EQUALIZATION_FILE], dicttype = Dict{String, Float64}))
    else
        dataset = EqualizedDataset(dataset)
    end

    if get(config, KEY_EQUALIZATION_SAVE, true) && haskey(config, KEY_EQUALIZATION_FILE) && (!isfile(config[KEY_EQUALIZATION_FILE]) || get(config, KEY_EQUALIZATION_FORCE_SAVE, false))
        YAML.write_file(config[KEY_EQUALIZATION_FILE], category_weights(dataset))
    end

    return dataset
end

process_dataset_equalization_config(::Nothing, dataset::AbstractDataset) = process_dataset_equalization_config(Dict(), dataset)

const KEY_DATASET_IMSEG = "imseg"
const KEY_DATASET_EXTENSIONS = "extensions"
const KEY_DATASET_FILTERS = "filters"
const KEY_DATASET_NORMALIZED = "normalization"
const KEY_DATASET_RESIZED = "resize"
const KEY_DATASET_SCALE = "scale"
const KEY_DATASET_EQUALIZED = "equalize"

const DATASET_CONFIG_PROVIDER_FUNCTION_MAP = Dict{String, Function}(
    KEY_DATASET_IMSEG => process_dataset_image_segmentation_config
)

const DATASET_CONFIG_EXTENSION_FUNCTION_MAP = Dict{String, Function}(
    KEY_DATASET_EXTENSIONS => process_dataset_extensions_config,
    KEY_DATASET_FILTERS => process_dataset_filters_config,
    KEY_DATASET_NORMALIZED => process_dataset_normalization_config,
    KEY_DATASET_RESIZED => process_dataset_resize_config,
    KEY_DATASET_SCALE => process_dataset_scale_config,
    KEY_DATASET_EQUALIZED => process_dataset_equalization_config
)

function register_dataset_config_provider(provider_name::AbstractString, config_processing_function::Function)
    DATASET_CONFIG_PROVIDER_FUNCTION_MAP[provider_name] = config_processing_function
end

function register_dataset_config_extension(extension_name::AbstractString, config_processing_function::Function)
    DATASET_CONFIG_EXTENSION_FUNCTION_MAP[extension_name] = config_processing_function
end

function process_dataset_config(config::AbstractDict)
    config_keys = collect(keys(config))

    config_provider_function_map = DATASET_CONFIG_PROVIDER_FUNCTION_MAP
    config_extension_function_map = DATASET_CONFIG_EXTENSION_FUNCTION_MAP

    provider_config_key = config_keys[findfirst(k -> haskey(config_provider_function_map, k), config_keys)]

    dataset = config_provider_function_map[provider_config_key](config[provider_config_key])

    for (config_key, extension_config) in config
        if config_key != provider_config_key && config_key != KEY_DATASET_EXTENSIONS && config_key != KEY_DATASET_FILTERS && haskey(config_extension_function_map, config_key)
            dataset = config_extension_function_map[config_key](extension_config, dataset)
        end
    end

    if KEY_DATASET_EXTENSIONS in keys(config)
        dataset = process_dataset_extensions_config(config[KEY_DATASET_EXTENSIONS], dataset)
    end

    if KEY_DATASET_FILTERS in keys(config)
        dataset = process_dataset_filters_config(config[KEY_DATASET_FILTERS], dataset)
    end

    return dataset
end

function process_dataset_config(configs)
    config_indices = eachindex(configs)

    dataset = DATASET_CONFIG_PROVIDER_FUNCTION_MAP[first(keys(configs[config_indices[1]]))](first(values(configs[config_indices[1]])))
    
    for i in 2:length(config_indices)
        dataset = DATASET_CONFIG_EXTENSION_FUNCTION_MAP[first(keys(configs[config_indices[i]]))](first(values(configs[config_indices[i]])), dataset)
    end

    return dataset
end