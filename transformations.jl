using Random


function readImage(img_path)
    
    "Reads image from given path and convert into Array form"
    
    img = load(img_path)
    img_CHW = channelview(img) # 3*375*500
    img_CHW = convert(Array{Float64}, img_CHW)
    
    return img_CHW
end



function randomUniformBetween(low, high)
    """
    Returns uniformly sampled random number between given range
    """
    return rand()*(high-low) + low
end



function distortion(image; alpha=1, beta=0)
    """
    Distory images by adjusting pixel values.
    """
    tmp = image.* alpha .+ beta
    tmp[tmp .< 0] .= 0
    tmp[tmp .> 1] .= 1
    
    return tmp
end


function  contrastDistortion(image;lower=0.5, upper=1.5)
    """
    Contrast is the difference in brightness between objects or regions
    
    """
    alpha = randomUniformBetween(lower,upper)
    return distortion(image,alpha=alpha)    
end

function brightnessDistortion(image; beta=32/255.)
    "
    Image result for what is brightness and contrast
    Brightness refers to the overall lightness or darkness of the image.
    "
    beta = randomUniformBetween(-beta,beta)
    return distortion(image,beta=beta)

end

function hueDistortion(image; adjust_factor = 18 / 255.)
    
    
    """HSV model stands for (Hue, Saturation, Value) model"""
    
    img = channelview(colorview(HSV, float.(image)))
    rate = randomUniformBetween(-adjust_factor,adjust_factor)

    # Hue distortion is appleied
    img[1,:,:] .+= rate

    #Back to RGB
    img =  channelview(colorview(RGB, img))
    img = Array{Float32}(img)
    return img
end

function  saturationDistortion(image; lower=0.5, upper=1.5)
    
    """HSV model stands for (Hue, Saturation, Value) model"""
    
    adjust_factor = randomUniformBetween(lower,upper)
    img = channelview(colorview(HSV, float.(image)))
    rate = randomUniformBetween(-adjust_factor,adjust_factor)

    # saturationDistortion appleied
    img[2,:,:] .*= rate

    #Back to RGB
    img =  channelview(colorview(RGB, img))
    img = Array{Float64}(img)
    return img

end

function flip_inefficient(image)

    h,w = size(image)

    flipped = zeros(h,w)
    flippedImage
    for r in 1:h
        for c in 1:w
            flipped[w-c+1,r]=image[r,c]
        end
    end

end

function flip_vectorized(image)
    """
    Vectorized Horizontal Flip Function
    """
    #img = channelview(image)
    return image[:,:,end:-1:1]
    
end
    


function horizontalFlips(img,bounding_boxes)
    """
    #Horizontally Flips the Image and bounding boxes
    """

    c,h,w = size(img)
    # returns channelview
    img = flip_vectorized(img)
    bounding_boxes[:,[1,3]] =  w .- bounding_boxes[:,[1,3]] 

    return img,bounding_boxes[:,[3,2,1,4]]

end



function transformation(image,boxes,labels, difficulties, split)
    """
    Make model more robust to various input object sizes and shapes

    Augmentation techniques:
    - Use the entire original input image
    - Sample a patch so that the minimum jaccard overlap with the objects is 0.1, 0.3,
    0.5, 0.7, or 0.9.
    – Randomly sample a patch. Size of each sampled patch is [0.1, 1] of the original image size, and the aspect ratio is between 1 and 2
    - After sampling step, image is resized to fixed size --> SSD300 utilizes 300*300 images
    - horizontally flipped with probability of 0.5,
    - Photometrics distortions 
        -Brightness
        -Contrast
        -Saturation
        -Hue

    - Zoom out -> to detect smaller size objects
    - Randomly place an image on canvas of 16x of the original image filled with mean values

    returns Transformed KnetArray of image with size ( 3,300,300)
    
    """
    
    #Imagenet mean and std
    mean = reshape([0.485, 0.456, 0.406],(3,1,1))
    std = reshape([0.229, 0.224, 0.225],(3,1,1))
    
    
    #averageImage = reshape([0.48501960903990504, 0.45795686011220893, 0.4076039332969516 ],(1,1,3))
    
    
    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    
    # Augmentation is only applied on Training
    if split == "TRAIN"
        
        # applies photometric distortion
        new_image = photometric_distortion(new_image)
        
        
        if rand() < 0.5
            new_image, new_boxes = expand(new_image, new_boxes, mean ; max_scale = 4)
        end
        
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels, new_difficulties)
        
        if rand() < 0.5
            #returns channelview
            new_image, new_boxes = horizontalFlips(new_image, new_boxes)
        end
    end
    
    # Applies for every Image
    
    
    # Bounding boxes are converted to percent coordinates in here     
    new_image, new_boxes = resize(new_image, new_boxes; dims=(300, 300))


    
    new_image = normalize(new_image, mean, std)
    #println("Lastly",new_boxes)
    return new_image, new_boxes, new_labels, new_difficulties
end

    


function resize(image, boxes; dims=(300, 300), return_percent_coords=true)
    
    """
    Resize Image and bounding boxes to 300,300
    Converts bounding boxes from pixel coordinates to scaled coordinates.

    """
    
    image  = colorview(RGB,image)
    
    # Resize image
    new_image = imresize(image, dims)
    # Resize bounding boxes
    h,w = size(image)
    h,w = Float16(h), Float16(w)
    
    # boxes, numberOfBoxes x 4  
    new_boxes = boxes ./ [w h w h]  # percent coordinates

    if !return_percent_coords
        new_dims = [dims[2] dims[1] dims[2] dims[1]]
        new_boxes = new_boxes .* new_dims
       
    end
    

    #new_boxes =  floor.(new_boxes)
    new_image = channelview(new_image)
    new_image = Array{Float64}(new_image)

    return new_image, new_boxes
end

function random_crop(image, boxes, labels, difficulties)
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.
    Note that some objects may be cut out entirely.
  
    :returns: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    #image = channelview(image)
    c,original_h, original_w  = size(image)

    # Keep choosing a minimum overlap until a successful crop is made
    while true
        # Randomly draw the value for minimum overlap
        min_overlap_choices = [0., .1, .3, .5, .7, .9, nothing]  # 'nothing' refers to no cropping
        
        min_overlap_idx =  rand(1:length(min_overlap_choices))
        min_overlap = min_overlap_choices[min_overlap_idx]
        
        # If not cropping
        if min_overlap === nothing
            return image, boxes, labels, difficulties
        end

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in 1:max_trials
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = randomUniformBetween(min_scale,1)
            scale_w = randomUniformBetween(min_scale,1)
            new_h = Integer(floor(scale_h * original_h))
            new_w = Integer(floor(scale_w * original_w))
            
            
            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if !((0.5 < aspect_ratio) &&  (aspect_ratio < 2))
                continue
            end

            # Crop coordinates (origin at top-left of image)
            left=1
            try
                left = rand(1 : original_w - new_w)
            catch ArgumentError
                left = 1
            end
            
            top=1
            try
                top = rand(1: original_h - new_h)
            catch ArgumentError
                top=1
            end
            
            right = left + new_w
            bottom = top + new_h
            crop = [left top  right  bottom]  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop, boxes)  # (1, n_objects), n_objects is the no. of objects in this image
           
            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if maximum(overlap) < min_overlap
                continue
            end
            
            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, 1:2] + boxes[:, 3:end]) ./2.  # (n_objects, 2)
            
            # Find bounding boxes whose centers are in the crop
            #element wise comparison returns BitArray by multiplying BitArray, I am creating a mask
            centers_in_crop = (bb_centers[:, 1] .> left) .* (bb_centers[:, 1] .< right) .* (bb_centers[:, 2] .> top) .* (
                    bb_centers[:, 2] .< bottom)   
            
            # If not a single bounding box has its center in the crop, try again
            if sum(centers_in_crop) == 0
                continue
            end
            
            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]
            

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, 1:2] = max.(new_boxes[:, 1:2], crop[:,1:2])  # crop[:2] is [left, top]
            new_boxes[:, 1:2] .-= crop[:,1:2]
            new_boxes[:, 3:end] = min.(new_boxes[:, 3:end], crop[:,3:end])  # crop[2:] is [right, bottom]
            new_boxes[:, 3:end] .-= crop[:,1:2]

            return new_image, new_boxes, new_labels, new_difficulties
            end
        end
    end



function expand(image, boxes, averageImage ; max_scale = 4)
    """
    Augmentation technique introduced by authors to have a zoom out effect.
    Randomly placing image on 4x scale of the original Image. ( In the paper its stated as 16x but
    many repos use 4x scale.
    """
    #image = channelview(image)
    c,original_h, original_w  = size(image)

    scale = randomUniformBetween(1, max_scale)
    new_h = Integer(floor(scale * original_h))
    new_w = Integer(floor(scale * original_w))

    expandedImage = zeros(c,new_h, new_w)

    expandedImage[1,:,:] .= fill(averageImage[1])
    expandedImage[2,:,:] .= fill(averageImage[2])
    expandedImage[3,:,:] .= fill(averageImage[3])

    left=2
    try
        left = rand(1 :  new_w- original_w)
    catch ArgumentError
        left = 1
    end 
    top=1
    try
        top = rand(1: new_h - original_h)
    catch ArgumentError
        top=1
    end

    # Crop coordinates (origin at top-left of image)
    right = left + original_w
    bottom = top + original_h
    expandedImage[:,top:bottom-1, left:right-1] .= image
    boxes .+= [left top left top]

    return expandedImage,boxes

end

function photometric_distortion(image)
    """
    Applies photometric distortions such as Brightness, Contrast, Saturation, Hue in random order.
    """

    distortions = [
        "Brightness",
        "Contrast",
        "Saturation",
        "Hue"]

    for dist_index in randperm(size(distortions,1))

        distortion = distortions[dist_index]
        if  distortion == "Hue"
            image = hueDistortion(image)

        elseif  distortion == "Brightness"
            image = brightnessDistortion(image)

        elseif distortion == "Saturation"
            image = saturationDistortion(image)

        elseif distortion == "Contrast"
            image = contrastDistortion(image;lower=0.5, upper=1.5)
        end

    return image
    end
end


function normalize(new_image, mean, std)
    """
    Normalization with Image Net mean and std.
    """
    new_image = new_image .- mean
    new_image = new_image ./ std
    return new_image
end