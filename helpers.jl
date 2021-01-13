include("config.jl")



function decimate(layer, m, atype )
    
    # Downsample by keeping every m'th value
    # Used Converting Fully Connected layer to Convolutional Layer
     
    #layer = Array(layer)
    for d in 1:ndims(layer)
        #println(d)
        if m[d] != nothing
            layer = selectdim(layer,d, 1:m[d]:size(layer)[d])
            #println(size(layer))
        end
    end
    
    return atype(layer)
end



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
        weights, bias = denseToConv(weights,bias)
        return atype(weights),atype(bias)    
    end
end
    
    
    


#= function decimate(layer, m, atype )
    
    # Downsample by keeping every m'th value
    # Used Converting Fully Connected layer to Convolutional Layer
     
    layer = Array(layer)
    for d in 1:ndims(layer)
        println(d)
        if m[d] != nothing
            layer = selectdim(layer,d, 1:m[d]:size(layer)[d])
            #println(size(layer))
        end
    end
    return atype(layer)
end
      =#       
# Converting to the outputs to the number of SSD predefined filter number 8372 
function expectedShape(layerlength, batch_size, expected_output)
    
    ret_shape = (layerlength/batch_size/expected_output)
    #println(ret_shape)
    #println("batch_size in es : ",batch_size)
    
    return (expected_output,Integer(ret_shape),batch_size)
    
    
end



