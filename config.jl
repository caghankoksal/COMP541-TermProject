
using LightXML
using Images, FileIO
using Knet
using IterTools
using Statistics
using ImageDraw;
using  Images;
using CUDA,ArgParse, DataStructures
using Statistics
using Dates

using Random
import Base: length, size, iterate, eltype, IteratorSize, IteratorEltype, haslength, @propagate_inbounds, repeat, rand, tail
import .Iterators: cycle, Cycle, take

atype = CUDA.functional() ? KnetArray{Float32} : Array{Float32}





# PATHS
# PATHS
const annotation_path_train_VOC2012 = "/datasets/pascal_voc2012/VOCdevkit/VOC2012/Annotations"
const images_path_train_VOC2012 = "/datasets/pascal_voc2012/VOCdevkit/VOC2012/JPEGImages"
const main_path_train_VOC2012 = "/datasets/pascal_voc2012/VOCdevkit/VOC2012/ImageSets/Main"

const trainval_VOC2012 = "$main_path_train_VOC2012/trainval.txt"
const train_VOC2012 = "$main_path_train_VOC2012/train.txt"
const validation_VOC2012 = "$main_path_train_VOC2012/val.txt"
                                      
const annotation_path_test_VOC2012 = "/datasets/pascal_voc2012/Test/VOCdevkit/VOC2012/Annotations"
const images_path_test_VOC2012 = "/datasets/pascal_voc2012/Test/VOCdevkit/VOC2012/JPEGImages"
const main_path_test_VOC2012 = "/datasets/pascal_voc2012/Test/VOCdevkit/VOC2012/ImageSets/Main"

const test_VOC2012 = "$main_path_test_VOC2012/test.txt"

const images_path_trainval_VOC2007 = "/kuacc/users/ckoksal20/VOCdevkit/VOC2007/JPEGImages"
const main_path_trainval_VOC2007 = "/kuacc/users/ckoksal20/VOCdevkit/VOC2007/ImageSets/Main"
const annotation_path_trainval_VOC2007 = "/kuacc/users/ckoksal20/VOCdevkit/VOC2007/Annotations"

const trainval_VOC2007 = "$main_path_trainval_VOC2007/trainval.txt"

const annotation_path_test_VOC2007 = "/kuacc/users/ckoksal20/VOC2007Test/VOCdevkit/VOC2007/Annotations"
const images_path_test_VOC2007 = "/kuacc/users/ckoksal20/VOC2007Test/VOCdevkit/VOC2007/JPEGImages"
const main_path_test_VOC2007 = "/kuacc/users/ckoksal20/VOC2007Test/VOCdevkit/VOC2007/ImageSets/Main"

const test_VOC2007 = "$main_path_test_VOC2007/test.txt"


class_to_index = Dict( (item) => i for (i,item) in enumerate(readlines("classes.txt")) )
index_to_class = Dict( i => item  for(i,item) in enumerate(readlines("classes.txt")))