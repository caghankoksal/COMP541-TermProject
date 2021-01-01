function createPolygons(bbxs)
    
    """
    To draw bounding boxes, Polygon object should be created
    
    Input : 2D Array of bounding boxes (number_of_bb, 4)
    
    Output: Array of Polygons
    
    """

    original_scale = 300
    
    #converts bounding boxes from percent coordinates to pixel coordinates
    bbxs = bbxs.*original_scale
    clamp!(bbxs,1,original_scale)
    bbxs = floor.(bbxs)
    polygons = []
    println(bbxs)
    
    numberOfBbx = size(bbxs,1)
    for i in 1:numberOfBbx
        box = bbxs[i,:]
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        leftTop = Point(xmin, ymin)
        leftbottom = Point(xmin,ymax)
        rightbottom = Point(xmax,ymax)
        rightTop = Point(xmax, ymin)
        curPoly = Polygon([leftTop,leftbottom,rightbottom,rightTop])
        push!(polygons,curPoly)
    end
    polygons
end