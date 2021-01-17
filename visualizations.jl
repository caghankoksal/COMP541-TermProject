function createPolygons(bbxs)
    
    """
    To draw bounding boxes, Polygon object should be created
    
    Input : 2D Array of bounding boxes (number_of_bb, 4)
    
    Output: Array of Polygons
    
    """
    #println("Incoming ",bbxs)
    original_scale = 300
    
    #converts bounding boxes from percent coordinates to pixel coordinates
    #bbxs = bbxs.*original_scale
    
    #clamp!(bbxs,1,original_scale)
    
    
    
    polygons = []
    #println("bbxs", size(bbxs) )
    numberOfBbx = size(bbxs,1)
    
    #println(size(bbxs))
    #println(bbxs)
    #println("number of box : ",numberOfBbx)
    for i in 1:numberOfBbx
        box = bbxs[i,:]
        #println("Current box before scale :: ",box)
        
        box = box.*original_scale
        box = clamp.(box,1,300)
        xmin = box[1]
        xmin = floor.(xmin)
        
        ymin = box[2]
        ymin = floor.(ymin)
        
        xmax = box[3]
        xmax = ceil.(xmax)
        
        ymax = box[4]
        ymax = ceil.(ymax)
        
        leftTop = Point(xmin, ymin)
        leftbottom = Point(xmin,ymax)
        rightbottom = Point(xmax,ymax)
        rightTop = Point(xmax, ymin)
        curPoly = Polygon([leftTop,leftbottom,rightbottom,rightTop])
        push!(polygons,curPoly)
    end
    polygons
end

"""

Inputs:

    index : Index of the image in batch. Mostly Integer between [1-32]
    x = Images in the batch
    all_images_boxes : Detected bounding boxes
    all_images_labels : Detected labels
    all_images_scores : Detected scores

Outputs:
    Image with bounding boxes

"""


function drawDetectedBoundingBox(index, x, all_images_boxes, all_images_labels, all_images_scores  )
    
    mean = Array(reshape([0.485, 0.456, 0.406],(1,1,3)))
    std = Array(reshape([0.229, 0.224, 0.225],(1,1,3)))
    
    ni2 = colorview(RGB, Array(permutedims(Array(x[:,:,:,index]).*std.+mean,(3,2,1))))
    
    #println(typeof(ni2))
    bboxes = all_images_boxes[index] #2-element Array{Any,1}: Float32[0.18513319 0.22615275 0.9629319 0.9930544]
    label_s = all_images_labels[index]
    conf_scores = all_images_scores[index]
    
    #println(label_s)
    #println("Size bboxex : ",size(bboxes))
    #println(conf_scores)
    
    
    ci = 1
    for i in 1:size(bboxes,1)
        
        curLabel = label_s[i][1]
        #println("label_s[i]", label_s[i])
        
        #println(i)
        if i>13
            ci = mod(i,13) +1
        end
        #println(i)
        color_text = nice_colors[ci]
        color = getColor(color_text)
        println("Color :  $color_text : " ,index_to_class[curLabel] )
        println("Confidence score : " ,conf_scores[i] )
        
        created_boxes = createPolygons(reshape(bboxes[i,:],(1,4)))
        for (k,pol) in enumerate(created_boxes)
            draw!(ni2,pol,RGB{Float64}(color...));
    
        end
        ci +=1

    end
        
   
return ni2
    
    
end
nice_colors = ["green" , "red", 
    "aqua", "blue","yellow","slateblue2", "violet","coral2","seagreen2", "deepskyblue1", "mediumblue",
    "purple1", "brown2"]

function getColor(color_name)
    return Colors.color_names[color_name]
end