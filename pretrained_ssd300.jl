function  pretrained_VGG_(pretrained_weights, train_the_weights = true)

    layer_1 = Chain([
        Conv(getweight_pytorch("base.conv1_1",pretrained_weights)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv1_2",pretrained_weights)..., f=relu, pool_ws = 2,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv2_1",pretrained_weights)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv2_2",pretrained_weights)..., f=relu, pool_ws = 2,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv3_1",pretrained_weights)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv3_2",pretrained_weights)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        
        #Conv(getweight_pytorch("base.conv3_3")..., f=relu, pool_ws = 2, pool_pad = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv3_3",pretrained_weights)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        PoolLayer(custom_padding = true),
        
        Conv(getweight_pytorch("base.conv4_1",pretrained_weights)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv4_2",pretrained_weights)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv4_3",pretrained_weights)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
    ])
    layer_2 = Chain([
        PoolLayer(window =2, stride=2, custom_padding = false ),
        Conv(getweight_pytorch("base.conv5_1",pretrained_weights)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv5_2",pretrained_weights)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getweight_pytorch("base.conv5_3",pretrained_weights)..., f=relu, pool_ws = 3, pool_stride = 1, pool_pad=1 ,trainable = train_the_weights),
        Conv(getweight_pytorch("base.conv6",pretrained_weights)..., pool_ws = 1, f=relu,pad=6, dilation=6,trainable = train_the_weights),
        Conv(getweight_pytorch("base.conv7",pretrained_weights)..., pool_ws = 1, f=relu, pad=0,trainable = train_the_weights)
        ])
    
    vgg = VGG16(layer_1,layer_2)
    return vgg
end



function pretrained_AuxiliaryHeads(pretrained_weights)
    layer1 = Chain([
        Conv(getweight_pytorch("aux_convs.conv8_1",pretrained_weights)...,  pad=0, pool_ws=1, f=relu),
        Conv(getweight_pytorch("aux_convs.conv8_2",pretrained_weights)...,  pad=1, conv_stride=2, pool_ws=1, f=relu)  # 10*10*512*1
    ])
    layer2 = Chain([
        Conv(getweight_pytorch("aux_convs.conv9_1",pretrained_weights)...,  pad=0, pool_ws=1, f=relu),  # dim. reduction because stride > 1
        Conv(getweight_pytorch("aux_convs.conv9_2",pretrained_weights)...,   pad=1, conv_stride=2, pool_ws=1, f=relu)  # dim. reduction because stride > 1
    ])
    layer3 = Chain([
        Conv(getweight_pytorch("aux_convs.conv10_1",pretrained_weights)..., pad=0,pool_ws=1,f=relu),
        Conv(getweight_pytorch("aux_convs.conv10_2",pretrained_weights)..., pad=0,pool_ws=1, f=relu)
    ])

    layer4 = Chain([
        Conv(getweight_pytorch("aux_convs.conv11_1",pretrained_weights)...,pool_ws=1, pad=0,f=relu),
        Conv(getweight_pytorch("aux_convs.conv11_2",pretrained_weights)...,pool_ws=1, pad=0,f=relu)

    ])

    return AuxiliaryHeads(layer1,layer2,layer3,layer4)
end


function pretrained_SSDheads(pretrained_weights;n_classes=21)
    
    localization_layers = Chain([
        Conv(getweight_pytorch("pred_convs.loc_conv4_3",pretrained_weights)...,  pool_ws=1, pad=1),
        Conv(getweight_pytorch("pred_convs.loc_conv7",pretrained_weights)..., pool_ws=1,pad=1),
        Conv(getweight_pytorch("pred_convs.loc_conv8_2",pretrained_weights)..., pool_ws=1, pad=1),
        Conv(getweight_pytorch("pred_convs.loc_conv9_2",pretrained_weights)..., pool_ws=1, pad=1),
        Conv(getweight_pytorch("pred_convs.loc_conv10_2",pretrained_weights)..., pool_ws=1, pad=1),
        Conv(getweight_pytorch("pred_convs.loc_conv11_2",pretrained_weights)..., pool_ws=1, pad=1)
    ])


    classification_layers = Chain([
        Conv(getweight_pytorch("pred_convs.cl_conv4_3",pretrained_weights)...,  pool_ws=1,pad=1),
        Conv(getweight_pytorch("pred_convs.cl_conv7",pretrained_weights)...,  pool_ws=1,pad=1),
        Conv(getweight_pytorch("pred_convs.cl_conv8_2",pretrained_weights)..., pool_ws=1, pad=1),
        Conv(getweight_pytorch("pred_convs.cl_conv9_2",pretrained_weights)...,  pool_ws=1, pad=1),
        Conv(getweight_pytorch("pred_convs.cl_conv10_2",pretrained_weights)..., pool_ws=1,  pad=1),
        Conv(getweight_pytorch("pred_convs.cl_conv11_2",pretrained_weights)..., pool_ws=1, pad=1)
    ])

    return SSDheads(localization_layers, classification_layers ,"",n_classes)

end


