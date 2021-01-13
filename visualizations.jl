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
    
    #println(bbxs)
    
    polygons = []
    
    numberOfBbx = size(bbxs,1)
    for i in 1:numberOfBbx
        box = bbxs[i,:]
        #println("BOX",box)
        
        box = box.*original_scale
        xmin = box[1]
        xmin = floor.(xmin)
        
        ymin = box[2]
        ymin = floor.(ymin)
        
        xmax = box[3]
        xmax = floor.(xmax)
        
        ymax = box[4]
        ymax = floor.(ymax)
        
        leftTop = Point(xmin, ymin)
        leftbottom = Point(xmin,ymax)
        rightbottom = Point(xmax,ymax)
        rightTop = Point(xmax, ymin)
        curPoly = Polygon([leftTop,leftbottom,rightbottom,rightTop])
        push!(polygons,curPoly)
    end
    polygons
end

function drawDetectedBoundingBox(index, x, all_images_boxes, all_images_labels, all_images_scores  )
    
    ni2 = colorview(RGB,(permutedims(Array(x[:,:,:,index] ),(3,2,1))))
    
    bboxes = all_images_boxes[index] #2-element Array{Any,1}: Float32[0.18513319 0.22615275 0.9629319 0.9930544]
    label_s = all_images_labels[index]
    conf_scores = all_images_scores[index]
    
    #println(label_s)
    #println(bboxes)
    println(conf_scores)
    for (i,bb) in enumerate(bboxes)
        
        curLabel = label_s[i][1]
        color_text = nice_colors[i]
        color = getColor(color_text)
        println("Color :  $color_text : " ,index_to_class[curLabel] )
        
        
        created_boxes = createPolygons(bb)
        for (k,pol) in enumerate(created_boxes)
            draw!(ni2,pol,RGB{Float32}(color...));
            
        end

    end
        
   
return ni2
    
    
end
nice_colors = ["green" , "red", 
    "aqua", "blue","yellow","slateblue2", "violet","coral2","seagreen2", "deepskyblue1", "mediumblue",
    "purple1", "brown2"]

function getColor(color_name)
    return Colors.color_names[color_name]
end