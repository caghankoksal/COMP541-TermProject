#= module VGG
using Knet,CUDA,ArgParse, DataStructures
include(Knet.dir("data","imagenet.jl"))

atype = CUDA.functional() ? KnetArray{Float32} : Array{Float32}

#const vggurl = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat"
#check vgg.jl for original version
const LAYER_TYPES = ["conv", "relu", "pool", "fc", "prob"]



using MAT,Images

function main()
    args = Dict()
    
    args["model"] = "/Users/caghankoksal/Desktop/COMP541/imagenet-vgg-verydeep-16"
    args["top"] = 5
    args["atype"] = CUDA.functional() ? KnetArray{Float32} : Array{Float32}
    
    atype = CUDA.functional() ? KnetArray{Float32} : Array{Float32}

    vgg = matread("../imagenet-vgg-verydeep-16.mat")
    (params),ldict = get_params(vgg, atype)
    
    description = vgg["meta"]["classes"]["description"]
    averageImage = convert(Array{Float32},vgg["meta"]["normalization"]["averageImage"])

    
    return vgg, params, description, averageImage, ldict
end

# This procedure makes pretrained MatConvNet VGG parameters convenient for Knet
# Also, if you want to extract features, specify the last layer you want to use
function get_params(CNN, atype; last_layer="prob")
    layers = CNN["layers"]
    weights, operations, derivatives = [], [], []
    
    
    ldict = OrderedDict()
    for l in layers
        get_layer_type(x) = startswith(l["name"], x)
        operation = filter(x -> get_layer_type(x), LAYER_TYPES)[1]
        push!(operations, operation)
        push!(derivatives, haskey(l, "weights") && length(l["weights"]) != 0)

        if derivatives[end]
            w = copy(l["weights"])
            if operation == "conv"
                w[2] = reshape(w[2], (1,1,length(w[2]),1))
            elseif operation == "fc"
                w[1] = transpose(mat(w[1]))
            end
            push!(weights, w)
            ldict[l["name"]] = w
        end

        last_layer != nothing && get_layer_type(last_layer) && break
    end

    (map(w -> map(wi->convert(atype,wi), w), weights), operations, derivatives), ldict
end


end =#



#using CUDA
#using PyCall
#@pyimport torch
#using Knet
#torch_model = torch.load("vgg16.bin")
#atype = CUDA.functional() ? KnetArray{Float32} : Array{Float32}

include("config.jl")


# Converts Pytorch Weights to the Knet Conv shape
function getweight(torch_model,layer_name;layer_type ="conv", atype = atype)
    
    weight_layer_name = "$layer_name.weight"
    bias_layer_name = "$layer_name.bias"
    weights = torch_model[weight_layer_name][:cpu]()[:numpy]()  # Return (n_output, input, kernel_size[0], kernel_size[1])
    bias = torch_model[bias_layer_name][:cpu]()[:numpy]()
    
    if layer_type == "conv"
        weights = permutedims(weights ,(3,4,2,1))
        n_output_bias = size(bias,1)
        bias = reshape(bias,(1,1,n_output_bias,1))
        return atype(weights),atype(bias)
        
    elseif layer_type=="dense"
        weights, bias = denseToConv_updated(weights,bias)
        return atype(weights),atype(bias)    
    end
end
    