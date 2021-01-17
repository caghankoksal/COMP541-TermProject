include("config.jl")


struct PoolLayer
    window
    padding
    stride
    mode # 0 for max, 1 for average including padded values, 2 for average excluding padded values, 3 for deterministic max.
    custom_padding
end

function PoolLayer(;window=2, padding = 0, stride = window, mode=0,custom_padding=false)
    PoolLayer(window, padding, stride, mode, custom_padding)
end


function (p::PoolLayer)(x) 
   if p.custom_padding
        pad_value = 0
        w,h,c,batch_size = size(x)
        
        t = atype(zeros(1, 1, c, batch_size))
        out = pool(cat(x, t; dims=(1,2)))

        
        return out
    end      
    return pool(x, window = p.window, padding = p.padding, stride = p.stride, mode=p.mode)
end
#(p::PoolLayer)(x) = pool(x, window = p.window, padding = p.padding, stride = p.stride, mode=p.mode)






function denseToConv(w,b)

    # Authors converts FC6 FC7 layers to CONV layers ;
    
    (n_output,n_input) = size(w)
    
    if size(w) == (4096,25088) #FC()
        
        w = reshape(permutedims(w),(7,7,512,4096))  
        w = decimate(w,[3,3,nothing,4],atype) # 3,3,512,1024 
        
        w = permutedims(w,(1,2,3,4))
        #w = reverse(w,dims=1)
        #w = reverse(w,dims=2)
        
        
        new_bias = decimate(b,[4],atype)
        new_bias = reshape(new_bias,(1,1,size(new_bias,1),1))
        
        new_bias = atype(new_bias)
        #return Conv_vgg(new_weights,new_bias,pad=6,dil=6,ws=1)
        return w, new_bias
        
    elseif size(w) == (4096,4096)
          
        w = reshape(permutedims(w),(1,1,4096,4096))  
        w = decimate(w,[nothing,nothing,4,4],atype) # 3,3,512,1024     
        w = permutedims(w,(1,2,3,4))
        #w = reverse(w,dims=1)
        #w = reverse(w,dims=2)
        
        w = atype(w)
        
        new_bias = decimate(b,[4],atype)
        new_bias = reshape(new_bias,(1,1,size(new_bias,1),1))
        
        new_bias = atype(new_bias)
        return w, new_bias
       
    end
    # reshape Dense to 
    #return Conv_vgg(new_weights,new_bias,ws=1,pad=0)
    
end


