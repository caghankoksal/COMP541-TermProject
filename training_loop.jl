#device_reset!()
include("vgg.jl")
include("layers.jl")
include("helpers.jl")
include("dataset.jl")
include("ssd4.jl")
include("utilities.jl")
include("train.jl")
include("transformations.jl")



function initopt!(model::SSD300, startLr = 1e-3,gclip = 0)
    for par in params(model)
        # Authors initialzied learning rates of bias 2x of weights learning rate
        if size(par)[1:2] == (1,1) && size(par)[end] == 1 #bias
            par.opt = Momentum(;lr=2*startLr, gclip=0, gamma = 0.9)
        else
            par.opt = Momentum(;lr=startLr, gclip=0,gamma = 0.9)
        end
    end
end
   
function currentTime()
    dt = now()
    return Dates.format(dt, "yyyy-mm-dd_HH:MM:SS")
end
function lrdecay!(model::SSD300, decay = 0.1)
    for param in params(model)
        param.opt.lr = param.opt.lr*decay
    end
end





batch_size = 16
train_dataset = PascalVOC(train_im_in_path, annotation_path, images_path,"TRAIN",dtype = atype, batchsize =batch_size)
validation_dataset = PascalVOC(val_im_in_path,annotation_path, images_path, "TEST", dtype=atype,batchsize=batch_size)
ssd300 = SSD300("CLUSTER")

iterations = 5000
epochs = round(iterations/length(train_dataset))


function currentTime()
    dt = now()
    return Dates.format(dt, "yyyy-mm-dd_HH:MM:SS")
end

