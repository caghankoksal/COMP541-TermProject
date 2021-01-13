function detect_objects(predicted_locs, predicted_scores; min_score = 0.2 , max_overlap = 0.5, top_k = 200)
    """
    predicted_locs :8732×4×32 KnetArray{Float32,3}:
    predicted_scores : 8732×21×32 KnetArray{Float32,3}:
    
    
    """
    #predicted_locs = permutedims(predicted_locs,(2,1,3))
    #predicted_scores = permutedims(predicted_scores,(2,1,3))
    #println("Number of classes",n_classes)
    predicted_locs = Array(predicted_locs)
    predicted_scores = Array(predicted_scores)
    predicted_scores = softmax(predicted_scores, dims=2)  
    batch_size = size(predicted_scores,3)
    
    priors_cxcy = Array(create_prior_boxes())
    all_images_boxes = []
    all_images_labels = []
    all_images_scores = []
    #println("min_score : ",min_score, "Max overlap : ",max_overlap)
    
    for i in 1:batch_size
        decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[:,:,i], priors_cxcy))  # (8732, 4), these are fractional pt. coordinates
        
        # Lists to store boxes and scores for this image
        image_boxes = []
        image_labels = []
        image_scores = []

        #max_scores, best_label = findmax(predicted_scores[:,:,i],dims=2)
        #println("max_scores",max_scores, "best_label : ",best_label)
        
        for c in 2:n_classes
        # Keep only predicted boxes and scores where scores for this class are above the minimum score
            class_scores = predicted_scores[:,c,i]  # 8732-element KnetArray{Float32,1}:
            
            #score_above_min_score = findall(x->x>score_above_min_score, class_scores)    
            score_above_min_score = class_scores .> min_score #8732-element BitArray{1}:
            n_above_min_score = sum(score_above_min_score)
            #println("N above : ",n_above_min_score, "class : ",c , "batch : ",i)
            if n_above_min_score == 0
                    continue
            end
           
            class_scores = class_scores[score_above_min_score] # (n_qualified), n_min_score <= 8732
            class_decoded_locs = decoded_locs[score_above_min_score,:]  # (n_qualified, 4) 127×4 Array{Float32,2}:
            
            #sort 
            sort_ind = sortperm(class_scores, rev=true) 
            
            class_scores = class_scores[sort_ind] #190-element Array{Float32,1}:
            class_decoded_locs = class_decoded_locs[sort_ind,:]
            
            #println("size class_decoded_locs", size(class_decoded_locs))
            # Find the overlap between predicted boxes
            overlap = find_jaccard_overlap_vectorized(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)
            
        
            suppress = zeros((n_above_min_score))
            
            
            for box in 1:size(class_decoded_locs,1) #127 -> n_qualified
                if suppress[box] == 1
                    #println("SUPPRESSED")
                    continue
                end
                #println(sum(suppress))
                suppress = max.(suppress, overlap[box,:] .> max_overlap )
                # Don't suppress this box, even though it has an overlap of 1 with itself
                suppress[box] = 0
            end
            #println("Number of supress: ", sum(suppress), " Number of box: ",size(class_decoded_locs,1))
            
            # Store only unsuppressed boxes for this class
            push!(image_boxes,class_decoded_locs[Bool.(1 .- suppress),:] )
            push!(image_labels,repeat([c],Int(sum(1 .- suppress))))
            push!(image_scores, class_scores[Bool.(1 .- suppress)])
            
            
        #println("image_boxes class",size(image_boxes))
        #println("image_labels class",size(image_labels))
        #println("image_scores class",size(image_scores))
            
        end
        
        
        
        #println(image_scores)
        # If no object in any class is found, store a placeholder for 'background'
        if length(image_boxes) == 0
            push!(image_boxes,[0 0 1 1])
            push!(image_labels,[0])
            push!(image_scores,[0])
        end
        
        
        #println("Size image boxes : ",size(image_boxes))
        #println("Size image_labels: ",size(image_labels))
        #println("Size image scores  : ",size(image_scores))
        
        #println(size(image_boxes[1]))
        
        
        # Concatenate into single tensors
        #println(size(image_boxes))
        image_boxes = vcat(image_boxes)  # (n_objects, 4)
        image_labels = vcat(image_labels)  # (n_objects)
        image_scores = vcat(image_scores)  # (n_objects)
        n_objects = size(image_scores,1)
        #println("N objects",n_objects)
        #println("Size image_scores",size(image_scores))
        
        #Keep only the top k objects
        if n_objects > top_k
            sort_ind = sortperm(image_scores, rev = true)
            image_scores = image_scores[sort_ind] 

            image_scores = image_scores[1:top_k]  # (top_k)
            image_boxes = image_boxes[sort_ind,:][1:top_k]  # (top_k, 4)
            image_labels = image_labels[sort_ind][1:top_k]  # (top_k)
        end
        # Append to lists that store predicted boxes and scores for all images
            
        push!(all_images_boxes, image_boxes)
        push!(all_images_labels, image_labels)
        push!(all_images_scores, image_scores) 
        
        
    end
        
    return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size
end
        

