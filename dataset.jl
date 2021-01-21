include("config.jl")

function readIndex(indices,annotation_pathD,images_pathD,split; augmentation = true, dtype=Array{Float32} )
    
    """
    Read the data with given indices.
    Converts Images to Knet Array on GPU.
    """
    x = []
    bounding_boxes = []
    labels = []
    difficulties = []
    
    
    #println("Annotation path", annotation_path)
    #println("images_path", images_path)
    
    #println("which dataset", whichDataset)
    
    #TODO parallel reading
    for idx in indices
        
        #for idx in indices
           
        curAnnotPath = annotation_pathD[idx]
        curImgPath = images_pathD[idx]
        bounding_box, label, diff= creteBBobject(curAnnotPath)
        
        # Reads Image
        img = readImage(curImgPath)
        
        new_image, new_boxes, new_labels, new_difficulties = transformation(img,bounding_box,label, diff, split)
        
        
        
        #new_image = reverse(new_image,dims=1)
        push!(x,new_image)
        push!(bounding_boxes,new_boxes)
        push!(labels,new_labels)
        push!(difficulties,new_difficulties)
    end
    x = cat(x...,dims=4)
    x = permutedims(x,(3,2,1,4))
    #println("Dtype ",dtype)
    x = dtype(x)
    return (x,bounding_boxes,labels,difficulties) 
end


"""
    PascalVOC(index, labels; batchsize::Int=8, shuffle::Bool=false)

Create a PascalVOC object, where `indices` is unique values for data points, 
`labels` is classes of bounding boxes,
`batchsize` is number of instances in a minibatch.
`shuffle` indicates shuffling the order of the instances before the whole iteration process.
"augmentation" indicates  applying  augmentation or not
"split" "TRAIN" or "TEST" if split is "TRAIN" --> augmentation is applied
"""

struct PascalVOC
    indices
    batchsize::Int
    shuffle::Bool
    num_instances::Int
    augmentation::Bool
    split
    annotation_path
    images_path
    dtype
    multi_dataset::Bool
    annotationsPathDict::Dict
    imagesPathDict::Dict
end
    
function PascalVOC(indices_path, annotation_path, images_path,split ; batchsize::Int=8, shuffle::Bool=false,
    augmentation = true,dtype::Type=Array{Float32}, multi_dataset = false)


    indices = []
    whichDataset = []
    
    annotationsPathDict = Dict()
    imagesPathDict = Dict()
    

    if indices_path == test_VOC2012
        xmlFiles = readdir(annotation_path)
        indices = [replace(el,".xml"=>"") for el in xmlFiles ]
        indices = filter!(e->e != [".2011_004297.xml.swp",".2011_004297.swp",".2011_004297.swp"],indices)

    elseif multi_dataset == true 
    # Assert whether indices_path, annotation_path, images_path == Array)

        for (i,p) in enumerate(indices_path)
            curIndices = readlines(p)
            push!(indices,curIndices )
            
            for eachIndex in curIndices
                
                curAnnotp = annotation_path[i]
                curImp = images_path[i]
                annotationsPathDict[eachIndex] =  "$curAnnotp/$eachIndex.xml"
                imagesPathDict[eachIndex] = "$curImp/$eachIndex.jpg"
            end
           

        end
        indices = vcat(indices...)
        

    else
        indices = readlines(indices_path)
        #ßprintln(indices)
        
        for eachIndex in indices
            annotationsPathDict[eachIndex] =  "$annotation_path/$eachIndex.xml"
            imagesPathDict[eachIndex] = "$images_path/$eachIndex.jpg"
        end
            
    end

    numInstance = size(indices,1)

    return PascalVOC(indices,batchsize,shuffle,numInstance,augmentation,split, annotation_path,
    images_path, dtype, multi_dataset,annotationsPathDict,imagesPathDict  )
end



function length(voc::PascalVOC)
    """
    # Returns number of batch in the dataset
    """
    numInstance = voc.num_instances
    bs = voc.batchsize
    return divrem(numInstance,bs)[2] != 0 ? length = divrem(numInstance,bs)[1] +1  : divrem(numInstance,bs)[1]
end


function iterate(data::PascalVOC, state=ifelse(
    data.shuffle, randperm(data.num_instances), collect(1:data.num_instances)))
    
    """
    Custom Iterator of the PascalVOC dataset, ignores the last batch,
    """

    bs = data.batchsize
    
    X = data.indices
    annotation_path = data.annotation_path
    img_path = data.images_path
        
    if size(state,1)== 0
        return nothing
    elseif size(state,1) < bs
        #(x,bounding_boxes,labels,difficulties)  = readIndex(X[state[1:end]], annotation_path, img_path, data.split, dtype = #data.dtype)
        #return ((x,bounding_boxes,labels), [])
        return nothing
    else
       
      (x,bounding_boxes,labels,difficulties) = readIndex(X[state[1:bs]],data.annotationsPathDict,data.imagesPathDict, data.split, dtype = data.dtype)
        return((x,bounding_boxes,labels,difficulties), state[bs+1:end])
    end
        
    
end



#