using CUDA

include("vgg.jl")
include("layers.jl")
include("helpers.jl")
include("dataset.jl")
include("ssd4.jl")
include("utilities.jl")
include("train.jl")
include("transformations.jl")
include("config.jl")
using Dates



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

function mytrain!(model::SSD300, train_data, test_data::PascalVOC,epochs
                  ;period::Int=2000, learning_rate_decay = 0.1 )
    
    train_loss = []
    test_loss = []
    iteration = 0
    curTime = currentTime()
    tr_loss = 0
    tst_loss = 0
    println("Training has started ", "length train_data : ",length(train_data), "Expected epochs : ",epochs)
    
    filename="log_ssd300_2012_VOC2007_2012trainval__l1_batch_$batch_size-iteration_$iterations-$curTime.txt"
    for epoch in 1:epochs
        start_epoch_time = now()
        println("Epoch : ", epoch, " has started")
       
        for (x,bounding_boxes,labels,_) in train_data
            
            if iteration == 80000|| (iteration > 80000 && iteration % 20000 == 0)

                lrdecay!(model,learning_rate_decay)

                println("Learning_rate decay is applied")
                curLr = params(model)[1].opt.lr
                open("/kuacc/users/ckoksal20/log_files/$filename", "a") do f
                    write(f, "Learning rate changed to $curLr epoch $epoch ")
                end
                curTime = currentTime()
                Knet.save("/kuacc/users/ckoksal20/trained_models/model_ssd300_l1_VOC2007_VOC2012trainval_cur_iteration-$iteration-$curTime.jld2","model_$iteration",model)
            end


            if iteration%period == 0
                tr_loss = 0
                tr_loss = model(train_data) 
                tst_loss = model(test_data)
                println("Iteration ", iteration, " training loss : ",tr_loss, " test_loss : ",tst_loss)
                open("/kuacc/users/ckoksal20/log_files/$filename", "a") do f
                    write(f, "Iteration: $iteration training loss $tr_loss test_loss $tst_loss \n")
                end
                append!(train_loss,tr_loss )
                append!(test_loss,tst_loss )
            end
            
            momentum!(ssd300,[(x,bounding_boxes,labels)])
            iteration +=1
        end
    end_epoch_time = now()
    elapsed_time = ((end_epoch_time -start_epoch_time).value)/1000/60
        
    println("Elapsed time $elapsed_time min epoch : $epoch\n")    
    open("/kuacc/users/ckoksal20/log_files/$filename", "a") do f
        write(f, "Elapsed time $elapsed_time min epoch : $epoch\n")
        #write(f,"Iteration: $iteration, training loss : $tr_loss, test_loss : $tst_loss \n")
    end
    end
        
    return 0:period:iteration, train_loss, test_loss
end






"""
The paper recommends training for 80000 iterations at the initial learning rate.
Then, it is decayed by 90% (i.e. to a tenth) for an additional 20000 iterations, twice. With the paper's batch size of 32, this means that the learning rate is decayed by 90% once after the 154th epoch and once more after the 193th epoch, and training is stopped after 232 epochs. I followed this schedule.
"""


batch_size = 32



train_dataset = PascalVOC(
    [trainval_VOC2007,trainval_VOC2012],
    [annotation_path_trainval_VOC2007,annotation_path_train_VOC2012],
    [images_path_trainval_VOC2007,images_path_train_VOC2012],"TRAIN",dtype = atype, batchsize =batch_size,
shuffle =true, multi_dataset = true)

test_VOC2007_dateset = PascalVOC(test_VOC2007, annotation_path_test_VOC2007, images_path_test_VOC2007,
    "TEST", dtype=atype,batchsize=batch_size, shuffle=true, multi_dataset = false)

ssd300 = SSD300(pretrained = false)
initopt!(ssd300)



iterations = 120000
epochs = round(iterations/length(train_dataset))

iters, trnloss, tstloss = mytrain!(ssd300,train_dataset,test_VOC2007_dateset,epochs)


