include("dataset.jl")
include("config.jl")
include("layers.jl")
include("pretrained_ssd300.jl")
include("utilities.jl")
include("train.jl")

n_boxes = Dict(
    "conv4_3" => 4,
    "conv7" => 6,
    "conv8_2" => 6,
    "conv9_2" => 6,
    "conv10_2" => 4,
    "conv11_2" => 4)

n_classes = 21

# Reading Pytorch VGG 
layer_names = Dict(
    "conv1_1" => "features.0",
    "conv1_2" => "features.2",
    "conv2_1" => "features.5",
    "conv2_2" => "features.7",
    "conv3_1" => "features.10",
    "conv3_2" => "features.12",
    "conv2_2" => "features.7",
    "conv3_1" => "features.10",
    "conv3_2" => "features.12",
    "conv3_3" => "features.14",
    "conv4_1" => "features.17",
    "conv4_2" => "features.19",
    "conv4_3" => "features.21",
    "conv5_1" => "features.24",
    "conv5_2" => "features.26",
    "conv5_3" => "features.28",
    "conv6" =>   "classifier.0",
    "conv7" =>   "classifier.3",
)




mutable struct Chain; layers; end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

mutable struct Conv
    w 
    b
    f
    pad
    conv_stride
    pool_window_size
    pool_stride#ps
    pool_pad
    dilation
    trainable
    atype
    end
function Conv(w1, w2, nx, ny; f=identity, pad=1, conv_stride =1, pool_ws=2,  pool_stride =pool_ws, pool_pad =0, dilation = 1, trainable = true, atype = atype)
    if trainable == true
        return Conv(param(w1,w2,nx,ny), param0(1,1,ny,1), f, pad, conv_stride , pool_ws,pool_stride, pool_pad, dilation , atype, trainable)
    else
        return Conv(Knet.value(param(w)), Knet.value(param(b)) ,f, pad, conv_stride , pool_ws,pool_stride, pool_pad, dilation , atype, trainable)
    end
end

function Conv(w,b; f=identity, pad=1, conv_stride =1, pool_ws=2,  pool_stride =pool_ws, pool_pad =0, dilation = 1, trainable = true, atype = atype)
    
    if trainable == true
        return Conv(param(w), param(b) ,f, pad, conv_stride , pool_ws,pool_stride, pool_pad, dilation , atype, trainable)
    else
        return Conv(Knet.value(param(w)), Knet.value(param(b)) ,f, pad, conv_stride , pool_ws,pool_stride, pool_pad, dilation , atype, trainable)
    end
end



(c::Conv)(x) = pool(c.f.(conv4(c.w, x, padding=c.pad, stride = c.conv_stride, dilation = c.dilation) .+ c.b), window = c.pool_window_size, padding=c.pool_pad, stride=c.pool_stride)
#(c::Conv)(x) = c.f.(conv4(c.w, x, padding=c.pad, stride = c.conv_stride, dilation = c.dilation, mode=0) .+ c.b)


mutable struct VGG16  firstlayers; secondlayers; end;


function VGG16(torch_model)

    layer_1 = Chain([
        Conv(getweight(torch_model, layer_names["conv1_1"])..., f=relu, pool_ws = 1,trainable = true )
        Conv(getweight(torch_model,layer_names["conv1_2"])..., f=relu, pool_ws = 2,trainable = true )
        Conv(getweight(torch_model,layer_names["conv2_1"])..., f=relu, pool_ws = 1,trainable = true )
        Conv(getweight(torch_model,layer_names["conv2_2"])..., f=relu, pool_ws = 2,trainable = false )
        Conv(getweight(torch_model,layer_names["conv3_1"])..., f=relu, pool_ws = 1,trainable = true )
        Conv(getweight(torch_model,layer_names["conv3_2"])..., f=relu, pool_ws = 1,trainable = true )
        Conv(getweight(torch_model,layer_names["conv3_3"])..., f=relu, pool_ws = 2, pool_pad = 1,trainable = true )
        Conv(getweight(torch_model,layer_names["conv4_1"])..., f=relu, pool_ws = 1,trainable = true )
        Conv(getweight(torch_model,layer_names["conv4_2"])..., f=relu, pool_ws = 1,trainable = true )
        Conv(getweight(torch_model,layer_names["conv4_3"])..., f=relu, pool_ws = 1,trainable = true )
    ])
    layer_2 = Chain([
        PoolLayer(0),
        Conv(getweight(torch_model,layer_names["conv5_1"])..., f=relu, pool_ws = 1,trainable = true ),
        Conv(getweight(torch_model,layer_names["conv5_2"])..., f=relu, pool_ws = 1,trainable = true ),
        Conv(getweight(torch_model,layer_names["conv5_3"])..., f=relu, pool_ws = 3, pool_stride = 1, pool_pad=1 ,trainable = true),
        
        Conv(getweight(torch_model,layer_names["conv6"],layer_type="dense")..., pool_ws = 1, f=relu,pad=6, dilation=6),
        Conv(getweight(torch_model,layer_names["conv7"],layer_type="dense")..., pool_ws = 1, f=relu, pad=0)
        ])

    return VGG16(layer_1,layer_2)
end


function(vgg::VGG16)(x)

    conv4_3_feats = vgg.firstlayers(x)
    conv7_feats = vgg.secondlayers(conv4_3_feats)

    return conv4_3_feats,conv7_feats
end




mutable struct AuxiliaryHeads layer1 ; layer2; layer3; layer4; end

function AuxiliaryHeads()
    layer1 = Chain([
        Conv(1,1,1024,256,  pad=0, pool_ws=1, f=relu),
        Conv(3,3, 256,512,  pad=1, conv_stride=2, pool_ws=1, f=relu)  # 10*10*512*1
    ])
    layer2 = Chain([
        Conv(1,1, 512,128, pad=0, pool_ws=1, f=relu),  # dim. reduction because stride > 1
        Conv(3,3, 128,256,  pad=1, conv_stride=2, pool_ws=1, f=relu)  # dim. reduction because stride > 1
    ])
    layer3 = Chain([
        Conv(1,1,256,128,pad=0,pool_ws=1,f=relu),
        Conv(3,3,128,256, pad=0,pool_ws=1, f=relu)
    ])

    layer4 = Chain([
        Conv(1,1,256,128,pool_ws=1, pad=0,f=relu),
        Conv(3,3,128,256,pool_ws=1, pad=0,f=relu)

    ])

    return AuxiliaryHeads(layer1,layer2,layer3,layer4)
end

function (aux_layers::AuxiliaryHeads )(x)

    out  = aux_layers.layer1(x)
    conv8_2_feats  = out

    out = aux_layers.layer2(out)
    conv9_2_feats = out

    out = aux_layers.layer3(out)
    conv10_2_feats = out

    out = aux_layers.layer4(out)
    conv11_2_feats = out

    return conv8_2_feats,conv9_2_feats,conv10_2_feats,conv11_2_feats
end


mutable struct SSDheads  
    localization_layers
    classification_layers 
    nboxes 
    n_classes
end




function SSDheads(nboxes,n_classes)

    localization_layers = Chain([
        Conv(3,3,512, n_boxes["conv4_3"] * 4,  pool_ws=1, pad=1),
        Conv(3,3,1024, n_boxes["conv7"] * 4, pool_ws=1,pad=1),
        Conv(3,3,512, n_boxes["conv8_2"] * 4, pool_ws=1, pad=1),
        Conv(3,3,256, n_boxes["conv9_2"] * 4, pool_ws=1, pad=1),
        Conv(3,3,256, n_boxes["conv10_2"] * 4, pool_ws=1, pad=1),
        Conv(3,3,256, n_boxes["conv11_2"] * 4, pool_ws=1, pad=1)
    ])


    classification_layers = Chain([
        Conv(3,3,512, n_boxes["conv4_3"] * n_classes,  pool_ws=1,pad=1),
        Conv(3,3,1024, n_boxes["conv7"] * n_classes,  pool_ws=1,pad=1),
        Conv(3,3,512, n_boxes["conv8_2"] * n_classes, pool_ws=1, pad=1),
        Conv(3,3,256, n_boxes["conv9_2"] * n_classes,  pool_ws=1, pad=1),
        Conv(3,3,256, n_boxes["conv10_2"] * n_classes, pool_ws=1,  pad=1),
        Conv(3,3,256, n_boxes["conv11_2"] * n_classes, pool_ws=1, pad=1)
    ])

    return SSDheads(localization_layers, classification_layers ,nboxes,n_classes)

end


function (prediction_heads::SSDheads)(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)
    
    batch_size = size(conv4_3_feats)[end]
    n_classes  = prediction_heads.n_classes
    #println("N_classes",n_classes)
    
    l_conv4_3 = prediction_heads.localization_layers.layers[1](conv4_3_feats)
    l_conv4_3 = permutedims(l_conv4_3,(3,1,2,4))
    #println("l_conv4_3 shape " ,size(l_conv4_3))
    l_conv4_3 = reshape(l_conv4_3,  expectedShape(length(l_conv4_3),batch_size,4))
    
    #l_conv4_3 = permuetdims(l_conv4_3, (3, 2, 1))
    
    #println(typeof(l_conv4_3))
    
  
    
    l_conv7 = prediction_heads.localization_layers.layers[2](conv7_feats)  # (19, 19, 24,1)
    l_conv7 = permutedims(l_conv7,(3,1,2,4))
    l_conv7 = reshape(l_conv7,  expectedShape(length(l_conv7),batch_size,4))
    
 
    
    #println("Conv 8_2 feats",size(conv8_2_feats))
    l_conv8_2 = prediction_heads.localization_layers.layers[3](conv8_2_feats)  # (10, 10, 24, 1)
    l_conv8_2 = permutedims(l_conv8_2,(3,1,2,4))
    #println("l_conv8_2 before reshape " ,size(l_conv8_2))
    l_conv8_2 = reshape(l_conv8_2, expectedShape(length(l_conv8_2),batch_size,4)  )
    #l_conv8_2 = permuetdims(l_conv8_2, (3, 2, 1))
    #println("l_conv8_2 after reshape " ,size(l_conv8_2))
    
    
    l_conv9_2 = prediction_heads.localization_layers.layers[4](conv9_2_feats)  # (5, 5, 24, 1)
    l_conv9_2 = permutedims(l_conv9_2,(3,1,2,4))
    l_conv9_2 = reshape(l_conv9_2, expectedShape(length(l_conv9_2),batch_size,4) )
    #l_conv9_2 = permuetdims(l_conv9_2, (3, 2, 1))
    
    
   
    l_conv10_2 = prediction_heads.localization_layers.layers[5](conv10_2_feats)  # (3, 3, 16, 1)
    l_conv10_2 = permutedims(l_conv10_2,(3,1,2,4))
    l_conv10_2 = reshape(l_conv10_2, expectedShape(length(l_conv10_2),batch_size,4) )
    
    
    l_conv11_2 = prediction_heads.localization_layers.layers[6](conv11_2_feats)  # (1, 1, 16, 1)
    l_conv11_2 = permutedims(l_conv11_2,(3,1,2,4))
    l_conv11_2 = reshape(l_conv11_2, expectedShape(length(l_conv11_2),batch_size,4) )
    
  


    # Predict classes in localization boxes
    c_conv4_3 = prediction_heads.classification_layers.layers[1](conv4_3_feats)  # (38, 38, 4 * n_classes, 1) --> (38, 38, 84, 1)
    c_conv4_3 = permutedims(c_conv4_3,(3,1,2,4))
    c_conv4_3 = reshape(c_conv4_3, expectedShape(length(c_conv4_3),batch_size,n_classes) )
    c_conv7 = prediction_heads.classification_layers.layers[2](conv7_feats)         # (19, 19, 6 * n_classes,1) -->  (19, 19, 126,1)
    c_conv7 = permutedims(c_conv7,(3,1,2,4))
    c_conv7 = reshape(c_conv7, expectedShape(length(c_conv7),batch_size,n_classes) )
    
    c_conv8_2 = prediction_heads.classification_layers.layers[3](conv8_2_feats)  # (10, 10, 6 * n_classes, 1) --> (10,10,24,1)
    c_conv8_2 = permutedims(c_conv8_2,(3,1,2,4))
    c_conv8_2 = reshape(c_conv8_2, expectedShape(length(c_conv8_2),batch_size,n_classes) )
    
    
    c_conv9_2 = prediction_heads.classification_layers.layers[4](conv9_2_feats)  # (5, 5, 6 * n_classes, 1)   --> (5,5,24,1)
    c_conv9_2 = permutedims(c_conv9_2,(3,1,2,4))
    c_conv9_2 = reshape(c_conv9_2, expectedShape(length(c_conv9_2),batch_size,n_classes) )
    
    c_conv10_2 = prediction_heads.classification_layers.layers[5](conv10_2_feats)  # ( 3, 3, 4 * n_classes,1)-->  (5,5,16,1)
    c_conv10_2 = permutedims(c_conv10_2,(3,1,2,4))
    c_conv10_2 = reshape(c_conv10_2, expectedShape(length(c_conv10_2),batch_size,n_classes) )
    
    
    c_conv11_2 = prediction_heads.classification_layers.layers[6](conv11_2_feats)  # ( 1, 1,4 * n_classes,1) --> (1,1,16,1)
    c_conv11_2 = permutedims(c_conv11_2,(3,1,2,4))
    
    c_conv11_2 = reshape(c_conv11_2, expectedShape(length(c_conv11_2),batch_size,n_classes) )    


                locs=hcat(l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2)  # (8732, 4,1)
    classes_scores = hcat(c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2)
        #,dims=2
                                 #)  # (8732, n_classes,1)
    
    classes_scores = permutedims(classes_scores,(2,1,3))
    locs = permutedims(locs,(2,1,3))
    
    #locs = permutedims(locs,(2,1,3))
    #classes_scores = permutedims(classes_scores,(2,1,3))
    #println("Locs shape ",size(locs))
    #println("classes_scores shape ",size(classes_scores))
    return locs,classes_scores
                
end

function getWeightFromKnetForFirstlayers(model::VGG16,index)
    
    w = Knet.value(model.firstlayers.layers[index].w) 
    b = Knet.value(model.firstlayers.layers[index].b)
    return w,b
end 

function getWeightFromKnetForSecondlayers(model::VGG16,index)
    
    w = Knet.value(model.secondlayers.layers[index].w) 
    b = Knet.value(model.secondlayers.layers[index].b)
    return w,b
    
end 

# Converts Pytorch Weights to the Knet Conv shape
function getweight_pytorch(layer_name,model;atype = atype)
    
    #
    #model = ssd_300_pretrained_weights["model"]
    
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


function getweight_rescale_pytorch(weights)
    
    #weights = model["rescale_factors"]
    n_channel = size(weights,2) #512
    res_factors = reshape(weights,(1,1,n_channel,1))
    return atype(res_factors)
end

    
    
    

function  CPU_weights_ToGpu(vgg16::VGG16, train_the_weights = true)

    layer_1 = Chain([
        Conv(getWeightFromKnetForFirstlayers(vgg16,1)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getWeightFromKnetForFirstlayers(vgg16,2)..., f=relu, pool_ws = 2,trainable = train_the_weights ),
        Conv(getWeightFromKnetForFirstlayers(vgg16,3)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getWeightFromKnetForFirstlayers(vgg16,4)..., f=relu, pool_ws = 2,trainable = train_the_weights ),
        Conv(getWeightFromKnetForFirstlayers(vgg16,5)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getWeightFromKnetForFirstlayers(vgg16,6)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getWeightFromKnetForFirstlayers(vgg16,7)..., f=relu, pool_ws = 2, pool_pad = 1,trainable = train_the_weights ),
        Conv(getWeightFromKnetForFirstlayers(vgg16,8)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getWeightFromKnetForFirstlayers(vgg16,9)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getWeightFromKnetForFirstlayers(vgg16,10)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
    ])
    layer_2 = Chain([
        PoolLayer(0),
        Conv(getWeightFromKnetForSecondlayers(vgg16,2)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getWeightFromKnetForSecondlayers(vgg16,3)..., f=relu, pool_ws = 1,trainable = train_the_weights ),
        Conv(getWeightFromKnetForSecondlayers(vgg16,4)..., f=relu, pool_ws = 3, pool_stride = 1, pool_pad=1 ,trainable = train_the_weights),
        Conv(getWeightFromKnetForSecondlayers(vgg16,5)..., pool_ws = 1, f=relu,pad=6, dilation=6,trainable = train_the_weights),
        Conv(getWeightFromKnetForSecondlayers(vgg16,6)..., pool_ws = 1, f=relu, pad=0,trainable = train_the_weights)
        ])
    
    vgg = VGG16(layer_1,layer_2)
    return vgg
end

function readBinaryTorchModel(model_path)

    torch = pyimport("torch")
    torch_model = torch.load(model_path)

    return torch_model

end


mutable struct SSD300 VGG; auxilaryLayer; heads; priors_cxcy; rescale_factor; platform;  end


function SSD300(;platform = "CLUSTER", pretrained = false)
    

    if pretrained == false
        if platform == "CLUSTER"
            # Knet model but CPU weights Array32 type
            vgg16 = Knet.load("/kuacc/users/ckoksal20/vgg16.jld2", "vgg16")
            # convert CPU weight to GPU weights
            vgg16 = CPU_weights_ToGpu(vgg16)
        elseif platform == "CPU"
            torch_model = readBinaryTorchModel("vgg16.bin")
            # Data Reading Code Should be added

            vgg16 = VGG16(torch_model)
        end

        auxHeads = AuxiliaryHeads()
        ssdHeads = SSDheads(n_boxes,n_classes)

        priors_cxcy = create_prior_boxes()
        rescale_factor = Float32(20.0)

        return SSD300(vgg16,auxHeads,ssdHeads, priors_cxcy,rescale_factor,platform)
        
        
    elseif pretrained== true
        
        println("Pretrained constructor is called")
        ssd_300_pretrained_weights = load("/kuacc/users/ckoksal20/ssd300_pretrained_weights.jld")
        ssd_300_pretrained_weights = ssd_300_pretrained_weights["model"]
        
        
        pretrained_vgg16 =  pretrained_VGG_(ssd_300_pretrained_weights)
        pretrained_auxHeads = pretrained_AuxiliaryHeads(ssd_300_pretrained_weights)
        pretrained_ssdHeads =  pretrained_SSDheads(n_classes = 21,ssd_300_pretrained_weights)
        
        #rescale_factor = Float32(mean(model["rescale_factors"]))
        
        rescale_factor = getweight_rescale_pytorch(ssd_300_pretrained_weights["rescale_factors"])
        priors_cxcy = create_prior_boxes()
        
        return SSD300(pretrained_vgg16, pretrained_auxHeads, pretrained_ssdHeads,priors_cxcy,rescale_factor,platform) 
        
    end
        
        
    
end


function(ssd300::SSD300)(x)

    conv4_3_feats, conv7_feats = ssd300.VGG(x)
    
    norm = (sum(conv4_3_feats.^2,dims=3).^(1/2))
    conv4_3_feats = conv4_3_feats ./ norm
    
    #println("size(conv4_3_feats",size(conv4_3_feats))
    #println("size(rescale_factor",size(ssd300.rescale_factor))
    conv4_3_feats = conv4_3_feats .* ssd300.rescale_factor
    
    

    conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = ssd300.auxilaryLayer(conv7_feats)

    locs, classes_scores = ssd300.heads(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)
    
    #println("size(conv4_3_feats",size(conv4_3_feats))
    ##println("size(conv7_feats",size(conv7_feats))
    #println("size(conv8_2_feats",size(conv8_2_feats))
    #println("size(conv9_2_feats",size(conv9_2_feats))
    #println("size(conv10_2_feats",size(conv10_2_feats))
    #println("size(conv11_2_feats",size(conv11_2_feats))


    return locs,classes_scores 
end

function weightDecayLoss(ssd300::SSD300, lambda = 5e-4 )
    weight_decay_loss = 0
    for  param in params(ssd300)
        #weight_decay_loss += lambda*sum((param.value).^2)
        weight_decay_loss += lambda*sum((param).^2)
    end
    return weight_decay_loss
    
end
function(ssd300::SSD300)(x,boxes,labels)

    predicted_locs,predicted_scores  = ssd300(x)
    #println("Predicted_locs_size",size(predicted_locs))
    #println("predicted_scores_size",size(predicted_scores))
        
    
    forward_loss = loss_forward(predicted_locs, predicted_scores, boxes, labels,atype,ssd300.priors_cxcy)
    weight_dec_loss = weightDecayLoss(ssd300)
    #println(weight_dec_loss)
    return weight_dec_loss + forward_loss
    #return forward_loss
end


function(ssd300::SSD300)(dataset::PascalVOC; iterate =5)
    
    numberOfBatch = length(dataset)
    
    loss = 0
    it = 1
    for (x,bounding_boxes,labels) in dataset
        loss+= ssd300(x,bounding_boxes,labels)
        
        if it == iterate
            break
        end
        it +=1
    end
    return loss/iterate
    
end



