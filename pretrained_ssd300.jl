# Converts Pytorch Weights to the Knet Conv shape
function getweight_pytorch(layer_name ;atype = atype)
    
    ssd_300_pretrained_weights = load("/kuacc/users/ckoksal20/ssd300_pretrained_weights.jld")
    model = ssd_300_pretrained_weights["model"]
   
    
    weight_layer_name = "$layer_name.weight"
    bias_layer_name = "$layer_name.bias"
    
    weights =  model[weight_layer_name]
    bias = model[bias_layer_name]

    weights = permutedims(weights ,(4,3,2,1))
    weights = reverse(weights,dims=1)
    weights = reverse(weights,dims = 2)

    
    n_output_bias = size(bias,1)
    bias = reshape(bias,(1,1,n_output_bias,1))
    return atype(weights),atype(bias)
        
end


function getweight_rescale_pytorch()
    ssd_300_pretrained_weights = load("/kuacc/users/ckoksal20/ssd300_pretrained_weights.jld")
    model = ssd_300_pretrained_weights["model"]
    weights  =  model["rescale_factors"]
    
    n_channel = size(weights,2) #512
    res_factors = reshape(weights,(1,1,n_channel,1))
    return atype(res_factors)
end


function getConvWeight(weights, atype)
    weights = permutedims(weights ,(4,3,2,1))
    
    
    weights = reverse(weights,dims=1)
    weights = reverse(weights,dims = 2)
    
    return atype(weights)
end

function getBiasWeight(weights, atype)
    n_output_bias = size(bias,1)
    bias = reshape(bias,(1,1,n_output_bias,1))
    atype(bias)
end



function  pretrained_VGG(train_the_weights = true)

    layer_1 = Chain([
        Conv(getweight_pytorch("base.conv1_1")..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv1_2")..., f=relu, pool_ws = 2,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv2_1")..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv2_2")..., f=relu, pool_ws = 2,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv3_1")..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv3_2")..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv3_3")..., f=relu, pool_ws = 2, pool_pad = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv4_1")..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv4_2")..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv4_3")..., f=relu, pool_ws = 1,trainable = train_the_weights ),
    ])
    layer_2 = Chain([
        PoolLayer(0),
        Conv(getweight_pytorch("base.conv5_1")..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv5_2")..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv5_3")..., f=relu, pool_ws = 3, pool_stride = 1, pool_pad=1 ,trainable = train_the_weights),
        Conv(getweight_pytorch("base.conv6")..., pool_ws = 1, f=relu,pad=6, dilation=6,trainable = train_the_weights),
        Conv(getweight_pytorch("base.conv7")..., pool_ws = 1, f=relu, pad=0,trainable = train_the_weights)
        ])
    
    vgg = VGG16(layer_1,layer_2)
    return vgg
end



function pretrained_AuxiliaryHeads()
    layer1 = Chain([
        Conv(getweight_pytorch("aux_convs.conv8_1")...,  pad=0, pool_ws=1, f=relu),
        Conv(getweight_pytorch("aux_convs.conv8_2")...,  pad=1, conv_stride=2, pool_ws=1, f=relu)  # 10*10*512*1
    ])
    layer2 = Chain([
        Conv(getweight_pytorch("aux_convs.conv9_1")...,  pad=0, pool_ws=1, f=relu),  # dim. reduction because stride > 1
        Conv(getweight_pytorch("aux_convs.conv9_2")...,   pad=1, conv_stride=2, pool_ws=1, f=relu)  # dim. reduction because stride > 1
    ])
    layer3 = Chain([
        Conv(getweight_pytorch("aux_convs.conv10_1")..., pad=0,pool_ws=1,f=relu),
        Conv(getweight_pytorch("aux_convs.conv10_2")..., pad=0,pool_ws=1, f=relu)
    ])

    layer4 = Chain([
        Conv(getweight_pytorch("aux_convs.conv11_1")...,pool_ws=1, pad=0,f=relu),
        Conv(getweight_pytorch("aux_convs.conv11_2")...,pool_ws=1, pad=0,f=relu)

    ])

    return AuxiliaryHeads(layer1,layer2,layer3,layer4)
end


function pretrained_SSDheads(;n_classes=21)
    
    localization_layers = Chain([
        Conv(getweight_pytorch("pred_convs.loc_conv4_3")...,  pool_ws=1, pad=1),
        Conv(getweight_pytorch("pred_convs.loc_conv7")..., pool_ws=1,pad=1),
        Conv(getweight_pytorch("pred_convs.loc_conv8_2")..., pool_ws=1, pad=1),
        Conv(getweight_pytorch("pred_convs.loc_conv9_2")..., pool_ws=1, pad=1),
        Conv(getweight_pytorch("pred_convs.loc_conv10_2")..., pool_ws=1, pad=1),
        Conv(getweight_pytorch("pred_convs.loc_conv11_2")..., pool_ws=1, pad=1)
    ])


    classification_layers = Chain([
        Conv(getweight_pytorch("pred_convs.cl_conv4_3")...,  pool_ws=1,pad=1),
        Conv(getweight_pytorch("pred_convs.cl_conv7")...,  pool_ws=1,pad=1),
        Conv(getweight_pytorch("pred_convs.cl_conv8_2")..., pool_ws=1, pad=1),
        Conv(getweight_pytorch("pred_convs.cl_conv9_2")...,  pool_ws=1, pad=1),
        Conv(getweight_pytorch("pred_convs.cl_conv10_2")..., pool_ws=1,  pad=1),
        Conv(getweight_pytorch("pred_convs.cl_conv11_2")..., pool_ws=1, pad=1)
    ])

    return SSDheads(localization_layers, classification_layers ,"",n_classes)

end



    
    
    