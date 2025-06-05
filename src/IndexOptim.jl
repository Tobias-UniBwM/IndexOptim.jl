module IndexOptim

include("utils.jl")
include("tvi.jl")
include("dataset.jl")
include("index.jl")
include("optim.jl")
include("config.jl")

include("CLI/CLI.jl")

@kwdef struct OptimizedIndexChannelExtension{TIndex <: AbstractOptimizedIndex} <: AbstractChannelExtension
    index::TIndex
    parameters::Vector{Float64}
    channel_order::Vector{String} = String[]
end

function compute_channel(extension::OptimizedIndexChannelExtension, capture::AbstractCapture)
    _channels = channels(capture)

    channel_order = extension.channel_order

    if isempty(channel_order)
        channel_order = sort!(collect(keys(_channels)))
    end

    images = map(n -> convert.(Float64, channelview(_channels[n])), channel_order)

    index_image = compute_index(extension.index, images, extension.parameters)
    normalize_unit!(index_image)
    
    return convert.(Gray, index_image)
end

const KEY_OPTIMIZED_INDEX_EXTENSION_INDEX = "index"
const KEY_OPTIMIZED_INDEX_EXTENSION_PARAMETERS = "parameters"
const KEY_OPTIMIZED_INDEX_EXTENSION_CHANNEL_ORDER = "channels"

const DEFAULT_VALUE_OPTIMIZED_INDEX_EXTENSION_CHANNEL_ORDER = String[]

function process_optimized_index_channel_extension_config(config::AbstractDict)
    index = process_optimized_index_config(config[KEY_OPTIMIZED_INDEX_EXTENSION_INDEX])
    parameters = config[KEY_OPTIMIZED_INDEX_EXTENSION_PARAMETERS]
    channel_order = get(config, KEY_OPTIMIZED_INDEX_EXTENSION_CHANNEL_ORDER, DEFAULT_VALUE_OPTIMIZED_INDEX_EXTENSION_CHANNEL_ORDER)

    return OptimizedIndexChannelExtension(index, parameters, channel_order)
end

const NAME_OPTIMIZED_INDEX_CHANNEL_EXTENSION = "opt_index"

function __init__()
    register_channel_extension_config_parser(NAME_OPTIMIZED_INDEX_CHANNEL_EXTENSION, process_optimized_index_channel_extension_config)
end

end # module IndexOptim
