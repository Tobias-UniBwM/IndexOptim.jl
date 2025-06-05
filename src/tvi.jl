using Statistics, Images

export 
    tvi

function tvi(image::AbstractArray{<:Real}, target_mask::AbstractArray{<:Bool}, unit_rescaling::Bool = true)
    image_tvi = unit_rescaling ? rescale_intensity(image) : image

    image_target = image_tvi[target_mask]
    image_background = image_tvi[.!target_mask]

    mean_target = mean(image_target)
    mean_background = mean(image_background)
    
    std_target = stdm(image_target, mean_target)
    std_background = stdm(image_background, mean_background)
    
    return tvi(mean_target, mean_background, std_target, std_background)
end

function w_tvi_image(image::AbstractArray)
    return convert.(Float64, image)
end

function w_tvi_image(image::AbstractArray{<:Colorant})
    return w_tvi_image(convert.(Gray, image))
end

function w_tvi_image(image::AbstractArray{<:Gray})
    return channelview(image)
end

function w_tvi_image(image::AbstractArray{<:Real})
    return image
end

function w_tvi_target_mask(target_mask::AbstractArray)
    return convert.(Bool, target_mask)
end

function w_tvi_target_mask(target_mask::AbstractArray{<:Colorant})
    return w_tvi_target_mask(convert.(Gray{Bool}, target_mask))
end

function w_tvi_target_mask(target_mask::AbstractArray{<:Gray{Bool}})
    return channelview(target_mask)
end

function w_tvi_target_mask(target_mask::AbstractArray{<:Bool})
    return target_mask
end

function tvi(image::AbstractArray, target_mask::AbstractArray, unit_rescaling::Bool = true)
    return tvi(w_tvi_image(image), w_tvi_target_mask(target_mask), unit_rescaling)
end

function tvi(mean_target::Real, mean_background::Real, std_target::Real, std_background::Real)
    return (mean_target - mean_background)^2 * (1 - 2 * std_target)^2 * (1 - 2 * std_background)^2
end

function w_tvi_value(value)
    return convert(Float64, value)
end

function w_tvi_value(value::Real)
    return value
end

function w_tvi_value(value::Gray)
    return gray(value)
end

function w_tvi_value(value::Colorant)
    return w_tvi_value(convert(Gray, value))
end

function tvi(mean_target, mean_background, std_target, std_background)
    return tvi(map(w_tvi_value, (mean_target, mean_background, std_target, std_background))...) 
end