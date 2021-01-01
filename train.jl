include("config.jl")


function create_prior_boxes()
    """
    Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
    By combining, prediction for all default boxes with different scales and aspect ratio,
    authors aims to cover various shaped objects.
    
    As authors states design of the optimal tiling open question and they used slightly different
    tiling than they mentioned on paper. 
    
    """
        #feature map dimensions
    fmap_dims = OrderedDict(
        "conv4_3" => 38,
        "conv7" => 19,
        "conv8_2" => 10,
        "conv9_2" => 5,
        "conv10_2" => 3,
        "conv11_2" => 1)

    # Prior box scales
    obj_scales = OrderedDict(
        "conv4_3" => 0.1,
        "conv7" => 0.2,
        "conv8_2" => 0.375,
        "conv9_2" => 0.55,
        "conv10_2" => 0.725,
        "conv11_2" => 0.9)

    #Predefined aspect ratio by authors
    aspect_ratios = OrderedDict(
        "conv4_3" => [1, 2, 0.5],
        "conv7" => [1, 2, 3, 0.5, 0.333],
        "conv8_2" => [1, 2, 3, 0.5, 0.333],
        "conv9_2" => [1, 2, 3, 0.5, 0.333],
        "conv10_2" => [1, 2, 0.5],
        "conv11_2" => [1, 2, 0.5])

    fmaps =fmap_dims.keys #"conv4_3" conv7""conv8_2" ,"conv9_2" "conv10_2" "conv11_2"
    prior_boxes = []
    additional_scale = 1
    for (k, fmap) in enumerate(fmaps) 
        for i in 1:fmap_dims[fmap]
            for j in 1:fmap_dims[fmap]
                #center of each box = (j + 0.5)/feature_map_size) , ( i+0.5) /feature_map_size 
                cx = (j-1 + 0.5) / fmap_dims[fmap]
                cy = (i-1 + 0.5) / fmap_dims[fmap]
                
                # width = scale * sqrt(aspect_ratio) 
                # height = scale/ sqrt(aspect_ratio) 
                for ratio in aspect_ratios[fmap]
                    push!(prior_boxes, [cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])
                    
                    # For the aspect ratio of 1, we also add a default box whose scale is sqrt(sk*sk+1)
                    if ratio==1
                        try
                        additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            catch
                            additional_scale = 1
                        end
                        push!(prior_boxes,[cx, cy, additional_scale, additional_scale])
                    end
                end
            end
        end
    end
    
    prior_boxes = permutedims(hcat(prior_boxes...),(2,1))
                                
    clamp!(prior_boxes, 0, 1)
    prior_boxes = atype(prior_boxes)
    return prior_boxes   
end


function find_intersection_vectorized(set_1, set_2)
    set1_num_obj = size(set_1,1)
    set2_num_obj = size(set_2,1)
    
    lower_bounds = max.(reshape(set_1[:,1:2], (set1_num_obj,1,2)),  reshape(set_2[:,1:2],(1,set2_num_obj,2)))
    upper_bounds = min.(reshape(set_1[:,3:end], (set1_num_obj,1,2)),  reshape(set_2[:,3:end],(1,set2_num_obj,2)))
    
    #intersections = clamp!(upper_bounds.-lower_bounds,0,1)
    intersections = max.(upper_bounds.-lower_bounds,0)
    
    return intersections[:,:,1] .* intersections[:,:,2]
    
    
end
function find_jaccard_overlap_vectorized(set_1, set_2)
    set1_num_obj = size(set_1,1)
    set2_num_obj = size(set_2,1)
    
    intersection = find_intersection_vectorized(set_1, set_2)  # (n1, n2)
    
    areas_set_1 = (set_1[:, 3] .- set_1[:, 1]) .* (set_1[:, 4] .- set_1[:, 2])  # (n1)
    areas_set_2 = (set_2[:, 3] .- set_2[:, 1]) .* (set_2[:, 4] .- set_2[:, 2])  # (n2)
    
    union = reshape(areas_set_1, (set1_num_obj, 1)) .+ reshape(areas_set_2, (1, set2_num_obj)) .- intersection
    return intersection ./ union #(n1,n2)
    
    
end



function smooth_l1_loss_index_vec(predicted_locs, true_locs,positive_indices)
    
    """
    
    Input : 
    
     #smootL1(x)      0.5x^2       abs(x) <1 
     #                abs(x)- 0.5  otherwise
    """

    loss = 0
    #predicted_locs2 = vec(predicted_locs)) #( 21,8372, 2)
    #true_locs2 = Array(Knet.value(true_locs))
    
    result = abs.(vec(predicted_locs)[positive_indices,:].-(vec(true_locs)[positive_indices,:])) 
     
    

    lower_indices = findall(x-> x<1, result)
    lower_loss = sum(0.5.*((result[lower_indices]).^2))
    loss += lower_loss   
    higher_indices =  findall(x-> x>=1, result)
    loss += sum(result[higher_indices] .-0.5)
    
    
     #return lower_indices, higher_indices
    return loss
end


function smooth_l1_loss_index(predicted_locs, true_locs,positive_indices)
    
    """
    Input predicted locations  (21,8372, batch_size)
    Input True Locations 
    Input : Positive Indices, where thei
    
     # function smootL1()
     # 0.5x^2   abs(x) <1 
     # abs(x)- 0.5  otherwise
    
    
    """
    
    loss = 0
    predicted_locs2 = Array(Knet.value(predicted_locs)) #( 21,8372, 2)
    true_locs2 = Array(Knet.value(true_locs))
    
    result = abs.(Array(predicted_locs2)[positive_indices,:].-(Array(true_locs2)[positive_indices,:])) 
     
    

    lower_indices = findall(x-> x<1, result)
    #lower_loss = sum(0.5.*((result[lower_indices]).^2))
    #loss += lower_loss   
    higher_indices =  findall(x-> x>=1, result)
    #loss += sum(result[higher_indices] .-0.5)
    
    
     return lower_indices, higher_indices
end

function smooth_l1_loss(predicted_locs, true_locs,positive_indices,lower_indices, higher_indices,n_positives)
    
    result = abs.(Array(predicted_locs)[positive_indices,:].-(Array(true_locs)[positive_indices,:])) 
    
    loss = 0
    lower_loss = sum(0.5.*((result[lower_indices]).^2)) / n_positives
    
    #println("Lower loss :",lower_loss)
    
    higher_loss = sum(result[higher_indices].-0.5) / n_positives
    
    #println("Higher loss :",higher_loss)
    
    return lower_loss + higher_loss
    
    
end

function l1_loss(predicted_locs, true_locs,positive_indices, posNumber)
    
     result = Array(predicted_locs)[positive_indices,:].-(Array(true_locs)[positive_indices,:]) 
     return sum(abs.(result))/posNumber
end

function loss_forward(predicted_locs, predicted_scores, boxes, labels,atype,priors_cxcy)
    #Forward propagation.
    """
    Input:
        predicted_locs: Output of model -predicted locations/boxes -  w.r.t the 8732 prior boxes, (N, 8732, 4)
        predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        
        boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        labels: true object labels, a list of N tensors
        atype : KnetArray if GPU else Float32Array
        priors_cxcy : Prior Boxes with center size coordinates.
    
        return:  Loss : Confidence Loss + Smooth L1 loss
        """
    
    batch_size = size(predicted_locs)[end]
    n_priors = size(priors_cxcy)[1]
    n_classes = size(predicted_scores)[2]
    neg_pos_ratio = 3
    threshold = 0.5
    #Convert priors from center size coordinates to boundary coordiantes
    priors_xy = cxcy_to_xy(priors_cxcy)
    
   
    true_locs = atype(zeros((n_priors, 4,batch_size))) # (N, 8732, 4)
    true_classes = atype(zeros((batch_size, n_priors))) # (N, 8732)
     
    
# For each image
    for i in 1:batch_size
        #println("Batch size",batch_size)
        # number of objects in image
        n_objects = size(boxes[i])[1]  # boxes Array store (nobject,dims) 2x4
        
        curBox = atype(boxes[i])
        #println(n_objects)
        overlap = find_jaccard_overlap_vectorized(curBox,priors_xy)  # (n_objects, 8732)
        
        #println("Overlap size",size(overlap))
        
        # For each prior, find the object that has the maximum overlap
        overlap_for_each_prior, object_for_each_prior = findmax(overlap,dims=1) 
        object_for_each_prior = map(i -> i[1],object_for_each_prior) # 1×8732 Array{Int64,2}:

        # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
        
        # To remedy this -
        # First, find the prior that has the maximum overlap for each object.
        _, prior_for_each_object = findmax(overlap,dims=2)  # (N_o)
            
        prior_for_each_object = map(i -> i[2],prior_for_each_object)
        
        # Connects these priors with the corresponding objects
        object_for_each_prior[:,prior_for_each_object] = 1:n_objects
        
        # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
        overlap_for_each_prior[:,prior_for_each_object] .= 1
        
        
        # Labels for each prior
        label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
        # Set priors whose overlaps with objects are less than the threshold to be background (no object)
        
        # Assign 0 to that labels, 
        # Eventough Julia has 1 based index system and background object is assigned to class number 1
        # For ease the NLL computations, I am assigning 0 in here.
        label_for_each_prior[overlap_for_each_prior .< threshold] .= 0  # (8732)
        
        
        # Store
        true_classes[i,:] = label_for_each_prior
       
        
        
        #ToDo : More efficent and clean code needed
        # For each bounding box that match with the prior we are extracting their coordinates
        to_encode = hcat([ boxes[i][j,:] for j in object_for_each_prior ]...)'
        
        # Encode center-size object coordinates into the form we regressed predicted boxes to
        offsets =  cxcy_to_gcxgcy(xy_to_cxcy(to_encode), priors_cxcy)  # (8732, 4)
        #println("offsets size",size(offsets))
        true_locs[:,:,i] = Array(offsets)
        
    end
    
    
    #println("Size true_classes ",size(true_classes))
    # Identify priors that are positive (object/non-background)
    
    true_classes = Array(true_classes)
    
    positive_indices = findall(x->x!=0, true_classes) # Cartesian Indices
    negative_indices = findall(x-> x==0, true_classes) 
    
    
    #println("Positive priors size",size(positive_priors))
    #println("Positive indices size",size(positive_indices))
    #println("Pos priors element",positive_priors[1])
    
    n_positives = size(positive_indices,1) #(42,1)
    
    
    
    predicted_locs = permutedims(predicted_locs,(3,1,2))
    true_locs = permutedims(true_locs,(3,1,2))
    
    
    # LOCALIZATION LOSS
    # Localization loss is computed only over positive (non-background) priors
    #loc_loss = l1_loss(predicted_locs,true_locs,positive_indices,n_positives )  # (), scalar
    
    #smooth l1 
    lower_indices, higher_indices = smooth_l1_loss_index(predicted_locs, true_locs,positive_indices)
    loc_loss = smooth_l1_loss(predicted_locs, true_locs,positive_indices,lower_indices,higher_indices,n_positives)
    #println("Localization loss", loc_loss)
    
    
# CONFIDENCE LOSS

    # Confidence loss is computed over positive priors and 
    # the most difficult (hardest) negative priors in each image(NEGATIVE HARD MINING)
    # For each image (neg_pos_ratio * n_positives) negative priors loss will be calculated.
    
    # Number of positive and hard-negative priors per image
    n_hard_negatives = neg_pos_ratio * n_positives  # (N)
    
    # Predicted_scores = (8732, 21, 10)
    predicted_scores = permutedims(predicted_scores,(2,1,3))      #(21, 8732, 10)
    
    true_classes = Array(true_classes)#True classes(2, 8732)
    
    #println("True classes",size(true_classes)) #True classes(2, 8732)
    #println("Predicted_scores",size(predicted_scores)) #Predicted_scores(21, 8732, 2)
    
    
    positiveIndices, hardestNegIndices, positiveTotal = findIndex_development(predicted_scores,true_classes;negativeRatio = neg_pos_ratio)
    confidenceLoss = calculateNLLoss(predicted_scores, true_classes, positiveIndices, hardestNegIndices, positiveTotal)
    
  
    conf_loss = confidenceLoss/n_positives
    total_loss = loc_loss  + conf_loss   
    
    
    println("Total loss : ",total_loss," Localization Loss : ",loc_loss, " Confidence Loss :",conf_loss)
    
    #println("Localization Loss : ",loc_loss)
return total_loss      
end

function calculateNLLoss(predicted_scores, true_classes, positiveIndices, hardestNegIndices, positiveTotal)
    """
    Inputs:
    True classes(batch_size, 8732)
    Predicted_scores(21, 8732, batch_size)
    hardestNegIndices : Hardest Negative Indices for Negative Hard Mining
    positiveTotal : Number of positive (non background) samples
    
    Output: Confidence Loss --> Positive Object loss + negative Loss
    """
    
    batch_size = size(true_classes,1)
  
    for i in 1:batch_size
        true_classes[i,hardestNegIndices[i]] .=1
    end
    
    #true_classes = permutedims(true_classes,(2,1))
    
    totalLoss = 0
    #            nll does not calculate loss on 0 indices, by converting hard negative indices to 1, loss can be calculated
    for i in 1:batch_size
        itemLoss = nll(predicted_scores[:,:,i], Integer.(true_classes[i,:]),dims=1,average=false) 
        totalLoss += itemLoss[1]
    end
    return totalLoss
end


function findIndex_development(predicted_scores,true_classes;negativeRatio = 3)
    
    """
    Find hardest negative indices by sorting the background scores for background object.
    Smallest Background scores for the background object will give the the most errorous classification for background.
    """

    true_classes = Array(true_classes) #(2,8372)
    predicted_scores2 = Array(Knet.value(predicted_scores)) #( 21,8372, 2)
    batch_size = size(true_classes,1)
  
    hardestNegIndices = []
    positiveIndices = []
    
    positiveTotal = 0
    for i in 1:batch_size
        #println(" TRUE CLASSES == 1", findall(x-> x==1, true_classes[i,:]))
        negative_indices = findall(x-> x==0, true_classes[i,:] )
        positive_indices = findall(x-> x!=0, true_classes[i,:]) # Cartesian Indices
        n_positive = size(positive_indices,1)
        n_negative = n_positive* negativeRatio
        
        positiveTotal+=n_positive
        #println("Negative indices",negative_indices)

        #println("Size negative at findIndex indices",size(negative_indices))

        negative_values = predicted_scores2[:,:,i][:,negative_indices]
        negative_values_scores_background = negative_values[1,:] # 1Background score
        #println("Size negative_values_scores_background",size(negative_values_scores_background))
        
        #Smallest Background scores for the background object will give the the most errorous classification for background.
        hard_indices =  sortperm(negative_values_scores_background)[1:n_negative]
        push!(hardestNegIndices,hard_indices)
        push!(positiveIndices,positive_indices)
    end
    
    return positiveIndices, hardestNegIndices,positiveTotal
    
end


