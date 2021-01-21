
function evaluate(test_dataset, model; min_score = 0.01, max_overlap = 0.45, top_k = 200)
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Lists to store detected and true boxes, labels, scores
    det_boxes = []
    det_scores = []
    det_labels= []
    true_boxes = []
    true_labels = []
    true_difficulties = [] 

   
        # Batches
        for (i, (images, boxes, labels, difficulties)) in enumerate(test_dataset)
            
            println(" Iteration : ",i)
            # Forward prop.
            predicted_locs, predicted_scores = model(images)
            
           
        
        
            # Detect objects in SSD output
        
            
            det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects(predicted_locs, 
            predicted_scores,
            min_score = min_score, 
            max_overlap = max_overlap,
            top_k = top_k)
        
     
            

            push!(det_boxes, det_boxes_batch)
            push!(det_labels, det_labels_batch)
            push!(det_scores, det_scores_batch)
            push!(true_boxes, boxes)
            push!(true_labels, labels)
            push!(true_difficulties,difficulties)
            
        
            #push!(true_difficulties, difficulties)
        end
        # Calculate mAP
        #det_boxes = cat(det_boxes...,dims=1)
        det_labels = cat(det_labels...,dims=1)
        det_scores = cat(det_scores...,dims=1)
        true_boxes = cat(true_boxes...,dims=1)
        true_labels = cat(true_labels...,dims=1)
        true_difficulties = cat(true_difficulties...,dims=1)
    
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels,true_difficulties)

    # Print AP for each class
    #println("Average Precisions : ", APs)

    #println("\nMean Average Precision (mAP): $mAP")
    
    return APs, mAP 
end

function calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
    n_classes = size(collect(class_to_index))[1]
    println("n_classes : ",n_classes)
    
    true_images = []
    for i in 1:size(true_labels,1)
        push!(true_images, repeat( [i],size(true_labels[i],1)))
    end
    true_images = vcat(true_images...) #(78,4)
    #println(true_images)
    println("Size true images",size(true_images))
    
    true_boxes = vcat(true_boxes...) #(78,4)
    true_labels = vcat(true_labels...)

    det_images = []
    for i in 1:size(det_labels,1)
        push!(det_images, repeat( [i],size(vcat(det_labels[i]...),1)))
    end
    det_images = vcat(det_images...)
    
    
    det_boxes = vcat(det_boxes...)
    det_boxes = vcat(det_boxes...)
    

    # 84×4 Array{Float32,2}:
    det_labels = vcat(det_labels...)
    det_labels = vcat(det_labels...) # 84-element Array{Int64,1}:
    
    det_scores = vcat(det_scores...)
    det_scores = vcat(det_scores...)
    true_labels = vcat(true_labels...)
    true_difficulties = vcat(true_difficulties...)
    
    println("Size true labels : ",size(true_labels))
    println("Size true_difficulties :", size(true_difficulties))
    println("Size det_scores : ",size(det_scores))
    println("Size det_labels : ",size(det_labels))
    println("Size det boxes :", size(det_boxes))
    
    
    #println("True labels ::",true_labels)
    
    average_precisions = zeros( n_classes -1)
    average_precisions_ = Dict()
    for c in 2:n_classes
        #println("Current class " , c)
        #To find which images have bounding boxes that have class c in their labels
        true_class_images = true_images[true_labels .== c]# 9-element Array{Int64,1}:
        
        #println("true_class_images: ", true_class_images)
        # Bounging boxes whose labels are class c.
        true_class_boxes = true_boxes[true_labels .== c,:]  # 9×4 Array{Float64,2}:  
        #println("True_class_boxes" ,true_class_boxes)
        #println("Size True class boxes", size(true_class_boxes))
        true_class_difficulties = true_difficulties[true_labels .== c]
        
        #println("true_class_difficulties : ",true_class_difficulties)
        n_easy_class_objects = 0
        try
            n_easy_class_objects = sum(1 .- true_class_difficulties)
        catch
            n_easy_class_objects = 0
        end
            
        
        #n_easy_class_objects = size((1 .- true_class_difficulties),1)
        #println("n_easy_class_objects : ",n_easy_class_objects)
            
        num_true_boxes = size(true_class_boxes,1)
        # Keep track of which true objects with this class have already been 'detected'
            # So far, none
        true_class_boxes_detected = zeros(size(true_class_boxes,1))  # (n_class_objects)


        # Extract only detections with this class
        det_class_images = det_images[det_labels .== c]  # (n_class_detections) 7-element Array{Int64,1}: 7 image contains  detection class c. 
        det_class_boxes = det_boxes[det_labels .== c,:]  # (n_class_detections, 4) 7×4 Array{Float32,2}:
        det_class_scores = det_scores[det_labels .== c]  # (n_class_detections)


        n_class_detections = size(det_class_boxes,1)
        #println("class : ",c, " n_class detections : ", n_class_detections, "det_class_boxes : ", size(det_class_boxes) )
        if n_class_detections == 0
            continue
        end

        # Sort detections by descending score
        sort_ind = sortperm(det_class_scores, rev=true) 

        #println("Sorted index",sort_ind)
        det_class_scores = det_class_scores[sort_ind]  #  n_class_detections) --> 7-element Array{Float32,1}:
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind,:]  # (n_class_detections, 4)
        true_positives = zeros(n_class_detections)
        false_positives = zeros(n_class_detections)

        for d in 1:n_class_detections
            this_detection_box = det_class_boxes[d,:] #4*1
            this_detection_box = reshape(this_detection_box, (1,4))

            # Corresponding image of the detection
            this_image = det_class_images[d] 
            # Find objects in the same image with the same class,  
            object_boxes = true_class_boxes[true_class_images .== this_image,:]
            object_difficulties = true_class_difficulties[true_class_images .== this_image] 
            #println(object_difficulties)
            
            # If that image does not have any bounding box --> There is false positive in here
            if size(object_boxes,1)== 0
                false_positives[d] = 1
                continue
            end

            #Finding Intersection over Union between current Detection and other ground truth objects in the image 
            overlaps = find_jaccard_overlap_vectorized(this_detection_box, object_boxes)  #--> (1, 3)
            
            # Since 
            max_overlap, ind = findmax(overlaps,dims=2)  # Returns CartesianIndex, 2nd dimension is the ground truth
            
            #println("Before ",ind)
            ind = map(i -> i[2],ind)
            #println( "After : ",ind)
            max_overlap = max_overlap[1]  #Extracting value from Array.
            #println("Size overlaps", size(overlaps), "max overlap :", max_overlap, "ind : ", ind) 
            #println("true_class_images", true_class_images)

            #println(collect(1:size(true_class_boxes,1)))
            original_ind = collect(1:size(true_class_boxes,1))[true_class_images .== this_image][ind][1]

            #println("Original index", original_ind)
            #println("Max overlap : ",max_overlap)
            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap > 0.5
                    # If this object has already not been detected, it's a true positive
                
                #println("object_difficulties[ind]", object_difficulties[ind])
                if object_difficulties[ind][1] == 0
                
            
                    #println(true_class_boxes_detected)
                    if true_class_boxes_detected[original_ind] == 0
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else
                        false_positives[d] = 1
                    end
                end
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else
                false_positives[d] = 1
            end
        end
        
         

            

           # println("True Positives :::",true_positives , "size : ", size(true_positives),"\n")
            #println("n_class_detections :",n_class_detections)
        
            cumul_true_positives = cumsum(true_positives, dims=1)  # (n_class_detections)
            cumul_false_positives = cumsum(false_positives, dims=1)  # (n_class_detections)

            #println("cumul_true_positives" , cumul_true_positives,"\n")
            #println("cumul_false_positives",cumul_false_positives)

            cumul_precision = cumul_true_positives ./(cumul_true_positives .+ cumul_false_positives .+ 1e-10)  # (n_class_detections)
            #cumul_recall = cumul_true_positives ./ num_true_boxes  # (n_class_detections)
            cumul_recall = cumul_true_positives ./ n_easy_class_objects
            recall_thresholds = collect(0:0.1:1)
        
            
            #println("Cumulative precision", cumul_precision ,"\n\n")
            #println("Cumulative recall",cumul_recall)
            #println("recall thresholds",recall_thresholds )
            precisions = zeros(size(recall_thresholds,1))
            #println("PREIZISISON",precisions)
            for (i,t) in enumerate(recall_thresholds)
                recalls_above_t = cumul_recall .>= t
                #println("recalls_above_t",recalls_above_t, "i",i)
                if sum(recalls_above_t) >0

                    #println("cumul_precision[recalls_above_t]",cumul_precision[recalls_above_t])
                    precisions[i] = maximum(cumul_precision[recalls_above_t])
                    
                else
                    precisions[i] = 0
                end
            end

            
        #println("Precisions : ",precisions)
        average_precisions[c - 1] = mean(precisions)
  
    end
    mean_average_precision = mean(average_precisions)
    println("mean_average_precision :: ", mean_average_precision)
    
    average_precisions_ = Dict( index_to_class[c + 1] => v for (c, v) in enumerate(average_precisions))
    
    println(average_precisions_)
    
    return mean_average_precision, average_precisions_
end

