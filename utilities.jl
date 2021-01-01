include("config.jl")
struct DataPoint
    image
    bounding_boxes
    DataPoint(image,bounding_boxes) = new(image,bounding_boxes)
end
struct BoundingBox 
    """ 
    Bounding Box Struct
    """
    class
    xmin
    ymin
    xmax
    ymax
    difficulty
    BoundingBox(class,xmin,ymin,xmax,ymax,difficulty) = new(class,xmin,ymin,xmax,ymax,difficulty)
end
# 
function IoU( ground_true_bb,predicted_bb)
    """
    Intersection over union (IoU) is a metric for measuring overlap between two bounding boxes
    """
    
    xmin_pred = predicted_bb[1]
    y_min_pred = predicted_bb[2]
    xmax_pred = predicted_bb[3]
    y_max_pred = predicted_bb[4]
    xmin_gt = ground_true_bb[1]
    y_min_gt = ground_true_bb[2]
    xmax_gt = ground_true_bb[3]
    y_max_gt = ground_true_bb[4]
    
    
    
    #  Proposed Intersection
    xmin_intersection = max(xmin_pred,xmin_gt ) # XminOfIntersection
    xmax_intersection = min(xmax_pred,xmax_gt)
    ymax_intersection = min(y_max_pred, y_max_gt )
    ymin_intersection = max(y_min_pred, y_min_gt)
    
    # Intersection Area
    interArea = max(0, xmax_intersection - xmin_intersection) * max(0, ymax_intersection - ymin_intersection )
    # totalArea =  PredBoxArea + #Grount TruthBox -IntersectionArea
    predBoxArea = abs(xmax_pred - xmin_pred) * abs(y_max_pred - y_min_pred)
    # Ground Truth Area
    gtBoxArea = abs(xmax_gt - xmin_gt) * abs(y_max_gt - y_min_gt)
    totalArea =  predBoxArea + gtBoxArea - interArea
    IoU = interArea / (totalArea)
    return IoU
end
function find_jaccard_overlap(boxes,priors)
    
    """
    Input : Prior boxes and Bounding Box array
    
    returns Array of IoU scores for each bounding box and prior pair.
    
    """
    
    allOverlap = []
    
    numberOfboxes = size(boxes,1)
    numberOfPriors = size(priors,1)
    
    overlap = zeros(numberOfboxes,numberOfPriors)
    for i in (1:numberOfboxes) # a list of N tensors
        for j in 1:numberOfPriors
            
            score = IoU(priors[j,:], boxes[i,:])
            
            overlap[i,j] = score
        end
        #push!(allOverlap,one_box)
    end

    return overlap
end


function boundingBoxToArray(bb)
    """
    # Takes BoundignBox returns coordinates as Array
    """
    xmin = bb.xmin
    ymin = bb.ymin
    xmax = bb.xmax
    ymax = bb.ymax
    return [xmin,ymin,xmax,ymax]
    
end


function creteBBobject(filename)
    """
    Reads and parse the XML file and create Bounding Box object.
    """
    xdoc = parse_file(filename)
    xroot = root(xdoc)  # an instance of XMLElement
    filename = content(xroot["filename"][1])
    bb_array = xroot["object"]
    boxes = []
    num_bb = size(bb_array,1)
    labels = []
    difficulties = []
    
    bounding_boxes = zeros(num_bb,4) # Xmin,ymin, xmax,ymax
    
    for (i,bb) in enumerate(bb_array)
        pose = content(find_element(bb, "pose"))
        class = content(find_element(bb, "name"))
        difficulty = content(find_element(bb, "difficult"))
        
        
        xmin = content(find_element(find_element(bb, "bndbox"),"xmin"))
        ymin = content(find_element(find_element(bb, "bndbox"),"ymin"))
        xmax  = content(find_element(find_element(bb, "bndbox"),"xmax"))
        ymax = content(find_element(find_element(bb, "bndbox"),"ymax"))
        
        
        # String to Integer
        x_min = parse(Int,xmin)
        x_max = parse(Int,xmax)
        y_min = parse(Int,ymin)
        y_max = parse(Int,ymax)
        
        difficulty = parse(Int,difficulty)  
        # classname --> integer mapping
        class_indx = class_to_index[class]
        
        # New Feature
        
        push!(difficulties,difficulty)
        push!(labels,class_indx)
        bounding_boxes[i,:] = [x_min,y_min,x_max,y_max]
        

    end
    free(xdoc)
    #return boxes
    return bounding_boxes,labels,difficulties

end





function gcxgcy_to_cxcy(gcxgcy, priors_cxcy)
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.
    They are decoded into center-size coordinates.
    This is the inverse of the function above.
    :gcxgcy: encoded bounding boxes, i.e. output of the model, (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    
    """
    
    # 10 and 5 can be interpreted as variances of the encoded bounding boxes
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    # Also see the blog post to understand why there is such division 
    # https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/

    centers = (gcxgcy[:, 1:2] .* priors_cxcy[:, 3:end] ./ 10) .+ priors_cxcy[:, 1:2]  # c_x, c_y
    width_height = exp.(gcxgcy[:, 3:end] ./ 5) .* priors_cxcy[:, 3:end]  # w, h
    
    return hcat(centers,width_height)
    
end
function cxcy_to_gcxgcy(cxcy, priors_cxcy)
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).
    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.
    
    In the model, we are predicting bounding box coordinates in this encoded form.
    
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # 10 and 5 can be interpreted as variances of the encoded bounding boxes
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    # Also see the blog post to understand why there is such division 
    # https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/
    
    
    priors_cxcy = Array(priors_cxcy)
    cxcy = Array(cxcy)
    centers = (cxcy[:, 1:2] .- priors_cxcy[:, 1:2]) ./ (priors_cxcy[:, 3:end]./ 10)  
    
    # TODO :
    width_height =    log.(cxcy[:, 3:end] ./ priors_cxcy[:, 3:end]) .* 5  # g_w, g_h
    #width_height =  (cxcy[:, 3:end] ./ priors_cxcy[:, 3:end]) .* 5  # g_w, g_h
    
    priors_cxcy = atype(priors_cxcy)
    cxcy = atype(cxcy)
    
    concat = hcat(centers, width_height)
    return atype(concat)
end

function cxcy_to_xy(cxcy)
    """
    Convert bounding boxes from center-size coordinates centerx, centery, width, height)
    to boundary coordinates  (xmin ymin xmax ymax)
    :cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    
    #println("Cxcy : ",cxcy[3:end])
    x_min_y_min = cxcy[:,1:2] .- (cxcy[:,3:end]./ 2 )  # x_min, y_min
    x_max_y_max = cxcy[:,1:2] .+ (cxcy[:,3:end] ./ 2)  # x_max, y_max
    
    return hcat(x_min_y_min, x_max_y_max)
    
end

function xy_to_cxcy(bbs)
    """
    Converts (xmin ymin xmax ymax) form to (centerx, centery, width, height) format 
    """

    centers = (bbs[:,1:2] .+ bbs[:,3:4])./2 #cx cy
    width_height = bbs[:,3:4] .- bbs[:,1:2] # width, height
    
    return hcat(centers,width_height)
    
end





    
