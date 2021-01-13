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
#println("typeof x :",typeof(x),"size x :",size(x))
   if p.custom_padding
        pad_value = 0
        w,h,c,batch_size = size(x)

        t = fill(pad_value, (w+1,h+1,c,batch_size) )
        t = atype(t)
        t[1:w, 1:h, :, : ]  = x
        
        #println("Type t",typeof(t))
        return pool(t)
    end      
    return pool(x, window = p.window, padding = p.padding, stride = p.stride, mode=p.mode)
end
#(p::PoolLayer)(x) = pool(x, window = p.window, padding = p.padding, stride = p.stride, mode=p.mode)


function custom_pool_pad(data)
    pad_value = 0
    w,h,c,batch_size = size(data)

    t = atype(fill(pad_value, (w+1,h+1,c,batch_size) ))
    t[1:w, 1:h, :, : ]  = data
    return pool(t)
    
end





"""

function custom_pool_pad(data)
    pad_value = -1000
    w,h,c,batch_size = size(data)

    t = atype(fill(pad_value, (w+1,h+1,c,batch_size) ))
    t[1:w, 1:h, :, : ]  = data
    return pool(t)
    
end

struct Pool_layer
    mode  # 0 for max, 1 for average including padded values, 2 for average excluding padded values, 3 for deterministic max.
    window #
    custom_pad
    pad_value
    
end
function Pool_layer(mode =0, window = 2, custom = true, pad_value = -1000)
    
    return Pool_layer(mode, window, custom,pad_value)
end
    

function (p::Pool_layer)(x)
    if p.custom_pad == true
        pad_value = p.pad_value
        w,h,c,batch_size = size(x)
        temp = atype(fill(pad_value, (w+2,h+2,c,batch_size) ))
        temp[2:w+1, 2:h+1, :, : ] = data
        return pool(t)
    else
        return pool(x,mode=0)
        end
end"""




function denseToConv(w,b)

    # Authors converts FC6 FC7 layers to CONV layers ;
    
    (n_output,n_input) = size(w)
    
    if size(w) == (4096,25088) #FC()
        
        w = reshape(permutedims(w),(7,7,512,4096))  
        w = decimate(w,[3,3,nothing,4],atype) # 3,3,512,1024        
        w = permutedims(w,(2,1,3,4))
        w = atype(w)
        
        new_bias = decimate(b,[4],atype)
        new_bias = reshape(new_bias,(1,1,size(new_bias,1),1))
        
        
        #return Conv_vgg(new_weights,new_bias,pad=6,dil=6,ws=1)
        return w, new_bias
        
    elseif size(w) == (4096,4096)
          
        w = reshape(permutedims(w),(1,1,4096,4096))  
        w = decimate(w,[nothing,nothing,4,4],atype) # 3,3,512,1024     
        w = permutedims(w,(1,2,4,3))
        new_bias = decimate(b,[4],atype)
        new_bias = reshape(new_bias,(1,1,size(new_bias,1),1))
        return w, new_bias
       
    end
    # reshape Dense to 
    #return Conv_vgg(new_weights,new_bias,ws=1,pad=0)
    
end



#= mutable struct Conv
    w 
    b
    f
    pad
    ws
    conv_strd
    end
Conv(w1, w2, nx, ny; f=sigm,pad=0,ws=2, conv_s =1) = Conv(param(w1,w2,nx,ny), param0(1,1,ny,1),f, pad, ws, conv_s)
(c::Conv)(x) = c.f.(pool(conv4(c.w, x,padding=c.pad, stride = c.conv_strd) .+ c.b,window=c.ws))


# Define dense layer:
mutable struct Dense; w; b; f; end
Dense(i,o; f=identity) = Dense(param(o,i;atype=atype), param0(o,atype=atype), f)
(d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b)

mutable struct Chain; layers; end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
#(c::Chain)(x,y) = nll(c(x),y)
#(c::Chain)(data::Knet.Train20.Data) = mean( c(x,y) for (x,y) in data)


# VGG convolutions --> non trainable
mutable struct Conv_vgg 
    w
    b
    f
    pad
    p_window_size
    ps # pool_stride
    dil
    pool_pad
    end
Conv_vgg(w; f=sigm,pad=1,ws=2,ps=ws,dil=1,pool_p=0)  = Conv_vgg(atype(w[1]),atype(w[2]),f,pad,ws,ps,dil,pool_p)
Conv_vgg(w,b; f=sigm,pad=1,ws=2,ps=ws,dil=1,pool_p=0)  = Conv_vgg(atype(w),atype(b),f,pad,ws,ps,dil,pool_p)
(c::Conv_vgg)(x) = c.f.(pool(conv4(c.w, x,padding=c.pad,mode=1,dilation = c.dil) .+ c.b, window = c.p_window_size, padding=c.pool_pad,stride=c.ps));
    
struct Dense_vgg
w
b
f
end
Dense_vgg(w; f=sigm)  = Dense_vgg(atype(w[1]),atype(w[2]),f)
(d::Dense_vgg)(x) = d.f.(  d.w * mat(x) + d.b);

struct PoolLayer
    mode
end
(p::PoolLayer)(x) = pool(x,mode=0)


function denseToConv(dense_vgg_layer::Dense_vgg)

    # Authors converts FC6 FC7 layers to CONV layers ;
    w = dense_vgg_layer.w
    b = dense_vgg_layer.b
    #println("Before b size",size(b))
    (n_output,n_input) = size(w)
    
    if size(w) == (4096,25088) #FC()
        new_weights = permutedims(reshape(dense_vgg_layer.w, (4096,512,7,7)),(3,4,2,1)) # shape (7,7,512,4096) 
        new_weights = decimate(new_weights, [3,3, nothing,4],atype )
        new_bias = decimate(b,[4,nothing],atype)
        new_bias = atype(reshape(Array(new_bias),(1,1,size(new_bias)[1],1)))
        return Conv_vgg(new_weights,new_bias,pad=6,dil=6,ws=1)
        
        
    elseif size(w) == (4096,4096)
        new_weights = permutedims(reshape(dense_vgg_layer.w,(4096,4096,1,1)),(3,4,2,1))  # (1,1,4096,4096)
        new_weights = decimate(new_weights, [nothing,nothing, 4,4],atype ) 
        new_bias = decimate(b, [4,nothing],atype)
        new_bias = atype(reshape(Array(new_bias),(1,1,size(new_bias)[1],1)))    
    end
    # reshape Dense to 
    return Conv_vgg(new_weights,new_bias,ws=1,pad=0)
end

# Extracting results of conv4_3 and conv7 layers
function vgg_semiLayerRepresentation(x, c::Chain)
    index = 1
    conv_4_3_output = []
    for l in c.layers
        x = l(x)
        if index == 10  #Conv4_3_layer_representation
            conv_4_3_output =x
        end
        index +=1
    end
    conv_7_output = x
    return (conv_4_3_output, conv_7_output)
end


 =#