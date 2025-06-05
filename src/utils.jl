using Images, Statistics

export 
    bitsizeof,
    pathext,
    read_file_paths,
    keys_as_symbols,
    normalize_linear,
    normalize_linear!,
    normalize_unit,
    normalize_unit!,
    normalize_z,
    normalize_z!,
    clean_data,
    clean_data!,
    @mustimplement

export 
    Optional,
    ImageValue,
    ImageGrayValue

const FILE_EXTENSION_DELIMITER = '.'

Optional{T} = Union{T, Nothing}
ImageValue = Union{Colorant, Real}
ImageGrayValue = Union{Gray, Real}

function bitsizeof(value)
    return sizeof(value) * 8
end

function pathext(path::AbstractString, ext::AbstractString, replace::Bool = true)
    return join((replace ? first(splitext(path)) : path, startswith(ext, FILE_EXTENSION_DELIMITER) ? ext : join((FILE_EXTENSION_DELIMITER, ext))))
end

function pathext(path::AbstractString, include_delimiter::Bool = true)
    return include_delimiter ? last(splitext(path)) : last(splitext(path))[begin + 1:end]
end

function read_file_paths(directory::AbstractString; include_subdirs::Bool = true, extensions = nothing)
    file_paths = String[]

    for file_path in readdir(directory, join = true, sort = true)
        if include_subdirs && isdir(file_path)
            append!(file_paths, read_file_paths(file_path, include_subdirs = include_subdirs, extensions = extensions))
        elseif isnothing(extensions) || !isempty(filter(e -> endswith(pathext(file_path), e), extensions))
            push!(file_paths, file_path)
        end
    end
    
    return file_paths
end

function keys_as_symbols(dict::AbstractDict{TKey, TValue}) where {TKey, TValue}
    converted_dict = Dict{Symbol, TValue}()

    for (key, value) in dict
        converted_dict[Symbol(key)] = value
    end

    return converted_dict
end

normalize_unit(data::AbstractArray{<:Integer}) = normalize_unit!(similar(data, Float64), data)
normalize_unit(data::AbstractArray) = normalize_unit!(similar(data), data)
normalize_unit!(data::AbstractArray) = normalize_unit!(data, data)
normalize_unit!(data_write::AbstractArray, data_read::AbstractArray) = normalize_linear!(data_write, data_read, (0., 1.))

normalize_linear(data::AbstractArray{<:Integer}, range::NTuple{2}) = normalize_linear!(similar(data, Float64), data, range)
normalize_linear(data::AbstractArray, range::NTuple{2}) = normalize_linear!(similar(data), data, range)
normalize_linear!(data::AbstractArray, range::NTuple{2}) = normalize_linear!(data, data, range)

function normalize_linear!(data_write::AbstractArray, data_read::AbstractArray, range::NTuple{2})
    min, max = extrema(data_read)

    data_write .= first(range) .+ (((data_read .- min) .* (last(range) - first(range))) ./ (max - min))
end

normalize_mean(data::AbstractArray{<:Integer}) = normalize_mean!(similar(data, Float64), data)
normalize_mean(data::AbstractArray) = normalize_mean!(similar(data), data)
normalize_mean!(data::AbstractArray) = normalize_mean!(data, data)

function normalize_mean!(data_write::AbstractArray, data_read::AbstractArray)
    _mean = mean(data_read)
    min, max = extrema(data_read)

    data_write .= (data_read .- _mean) ./ (max - min)
end

normalize_z!(data::AbstractArray; corrected_std::Bool = true) = normalize_z!(data, data, corrected_std = corrected_std)

function normalize_z!(data_write::AbstractArray, data_read::AbstractArray; corrected_std::Bool = true)
    _mean = mean(data_read)
    _std = std(data_read, mean = _mean, corrected = corrected_std)

    data_write .= (data_read .- _mean) ./ _std
end

normalize_z(data::AbstractArray, mean::Union{Real, AbstractArray}, std::Union{Real, AbstractArray}) = normalize_z!(copy(data), mean, std)

function normalize_z!(data::AbstractArray, mean::Union{Real, AbstractArray}, std::Union{Real, AbstractArray})
    data .= (data .- mean) ./ std
end

function normalize_z(data::AbstractArray, dims::Union{Integer, Tuple{Vararg{<:Integer}}, AbstractRange, Colon} = :; corrected_std::Bool = true)
    _mean = mean(data, dims = dims)
    _std = std(data, mean = _mean, dims = dims, corrected = corrected_std)
    
    return (data = normalize_z(data, _mean, _std), mean = _mean, std = _std)   
end

"""
The Graphics.jl package is licensed under the MIT "Expat" License:

> Copyright (c) 2015:
>  * Viral B. Shah
>
> Permission is hereby granted, free of charge, to any person obtaining
> a copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
> IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
> CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
> TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
> SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
macro mustimplement(sig)
    fname = sig.args[1]
    arg1 = sig.args[2]
    if isa(arg1,Expr) && arg1.head == :parameters && length(sig.args) > 2
        arg1 = sig.args[3]
    end
    if isa(arg1,Expr)
        arg1 = arg1.args[1]
    end
    :($(esc(sig)) = error(typeof($(esc(arg1))),
                          " must implement ", $(Expr(:quote,fname))))
end