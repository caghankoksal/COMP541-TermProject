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
    
    return layer
end


     
# Converting to the outputs to the number of SSD predefined filter number 8372 
function expectedShape(layerlength, batch_size, expected_output)
    
    ret_shape = (layerlength/batch_size/expected_output)
    #println(ret_shape)
    #println("batch_size in es : ",batch_size)
    
    return (expected_output,Integer(ret_shape),batch_size)
    
    
end



