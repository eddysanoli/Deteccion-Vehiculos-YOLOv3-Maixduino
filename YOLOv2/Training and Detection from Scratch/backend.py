## --------------------------------------------------------------------------------- ##
## Sample extraction: Extract a sample of images from a dataset
## --------------------------------------------------------------------------------- ##

import os
import shutil
import numpy as np

def clean_directory(path):
    """
    Clear the contents of a given directory or create it if it doesn't exist.

    Parameters
    -------------
    
    path: 
        Directory to clean 
    
    """

    print("Directory: ", path)
    success = True

    if os.path.exists(path):                                                # If: Path given exists                        

        if len(os.listdir(path)) == 0:                                      # If the directory is empty, the program deletes the directory
            print("No further action needed. Directory already exists.")     
            
        else:                                                               # If the directory is not empty the user its asked for permission to delete the directory
            option = input("Directory is not empty. Want to delete its contents? (y/n): ")

            if option == "y" or option == "Y":                              # Option selected is "yes"               
                try:
                    shutil.rmtree(path)                                     # Try to delete the directory and all of its contents
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))     # Return an error if any pop-up
                    
                os.mkdir(path)                                              # Then create the desired directory
           
            else:
                print("Directory not deleted")
                success = False
    
    else:
        print("Directory doesn't exist. It has been created.")
        os.mkdir(path)                                                      # If path doesn't exist, it is created.

    return success

def extract_sample(ann_dir, target_ann_dir, img_dir, target_img_dir, batch_size):
    """
    Extract a small batch of images and annotations from a dataset. Paths should
    have the form "folder/folder/folder/"

    Parameters
    -------------
    
    ann_dir: 
        Base directory where all the annotations from the dataset are stored
    
    target_ann_dir: 
        Target directory where the sample size of annotations will be stored
    
    img_dir: 
        Base directory where all the images from the dataset are stored
    
    target_img_dir: 
        Target directory where the sample size of images will be stored
    
    batch_size:
        Number of images/annotations pairs to extract from the dataset

    """

    # Program checks every element in the directory "img_dir"
    # If said element is a file, it is concatenated at the end of a list
    # len() then returns the lenght of the list. 
    image_count = len([name for name in os.listdir(img_dir) if os.path.isfile(img_dir + "/" + name)]) 

    step = int(np.floor(image_count / (batch_size + 1)))                    # Number of images that the program skips (Starting from image 0) to return the amount of images desired

    print("")
    print("Directory Cleaning")                                             # The target image and annotation directory is cleaned if needed

    image_clear = clean_directory(target_img_dir)                                      
    ann_clear = clean_directory(target_ann_dir)

    if image_clear and ann_clear:
        print("")
        print("Batch Creation")

        image_names = os.listdir(img_dir)                                   # A vector containing the name of all the files inside de "image" and "annotation" directories is saved
        ann_names = os.listdir(ann_dir)
        file_Count = 0

        for i in range(0, image_count + 1, step):                           # "i" ranges from 0 to ("image_count"+1) advancing with steps "step"

            # Both the annotations and images from the batch are
            # copied to the desired "target" directory
            shutil.copy(img_dir + image_names[i], target_img_dir + image_names[i]) 
            shutil.copy(ann_dir + ann_names[i], target_ann_dir + ann_names[i])

            file_Count += 1                                                 # "file_count" counts the number of pairs (image/annotation) that have been copied

            if file_Count % (batch_size / 10) == 0:                         # If the amount of files copied is a multiple of the 10th part of the batch_size
                print("Progress: ", (file_Count/batch_size) * 100, "%")     # a progress notification appears

            if file_Count == batch_size:
                break

        print("Batch creation finished.")
        print("")
    
    else:

        print("Batch creation haulted.")
        print("")

## --------------------------------------------------------------------------------- ##
## Parse Annotation: Extract images and labels from a group of annotations
## --------------------------------------------------------------------------------- ##

import xml.etree.ElementTree as ET
import os

def parse_annotations(ann_dir, img_dir, labels=[]):
    '''
    Parse XML files containing the annotations for the boundary boxes and extract all of its
    characteristics: Coordinates (X min / X max / Y min / Y max), file name, class label, width
    and height.

    Parameters
    ------------
    
    ann_dir (path structure = "folder/folder/")
        Directory containing all the training annotations (XML's)

    img_dir (path structure = "folder/folder/")
        Directory containing all the training images.

    labels (string list)
        List containing all the classes present on the dataset

    Returns
    ---------

    train_image (list of dictionary objects)
        List where each element consists of the relevant annotation data for one
        image.
        
    seen_train_labels (dictionary)
        Dictionary containing the items "key" (Object class) and "value" (Amount 
        of objects in class)
    '''
    all_imgs = []
    seen_labels = {}
    
    for ann in sorted(os.listdir(ann_dir)):                                             # Iterates through all the annotations in the directory                           
        if "xml" not in ann:                                                            # If the name of the file does not contain the suffix "xml" the program goes to the next file           
            continue
        

        img = {'object':[]}                                                             # Dictionary called image which contains a parameter "object"
        tree = ET.parse(ann_dir + ann)                                                  # The structure of the XML is stored in "tree"

        for elem in tree.iter():                                                        # The program parses through all the "tags" or parameters in the XML file.

            if 'filename' in elem.tag:                                                  # If the tag "filename" exists
                path_to_image = img_dir + elem.text                                     #       Text inside "filename" (elem.text) is stored in "path_to_image"
                img['filename'] = path_to_image                                         #       A new element of the dictionary "img" is added
                if not os.path.exists(path_to_image):                                   #       Make sure the file in "filename" exists
                    assert False, "file does not exist!\n{}".format(path_to_image)      #       If it doesnt exist, the program stops and returns an error
            
            if 'width' in elem.tag:                                                     # Save the "width" of the boundary box as an int
                img['width'] = int(elem.text)
            
            if 'height' in elem.tag:                                                    # Save the "height" of the boundary box as an int 
                img['height'] = int(elem.text)

            if 'object' in elem.tag or 'part' in elem.tag:                              # If the tag "object" or "part" exists
                
                obj = {}
                
                for attr in list(elem):                                                 #       The parser goes through every element in the "object" or "part" tag

                    if 'name' in attr.tag:                                              #       If the attribute "name" is found
                        
                        obj['name'] = attr.text                                         #       The "name" tag is stored inside the atribute "name" in "obj"
                        
                        if len(labels) > 0 and obj['name'] not in labels:               #       If the "label" list is not empty and the text inisde the "name" tag is not present inside said list
                            break                                                       #       The for loop is broken
                        else:                                                           #       Else, the "obj" object ({'name': objectname}) is added to the "object" parameter in "img" 
                            img['object'] += [obj]
                        
                        if obj['name'] in seen_labels:                                  #       If the classes detected in the image have already appeared in other XML's    
                            seen_labels[obj['name']] += 1                               #       The parser counts each instance of each class
                        else:                                                           #       If the class has not appeared
                            seen_labels[obj['name']]  = 1                               #       The counter for said class is set to 1
                            
                    if 'bndbox' in attr.tag:                                            # If the tag "bndbox" exists
                        for dim in list(attr):
                            if 'xmin' in dim.tag:                                       # The parser stores the "xmin", "xmax", "ymin" and "ymax" coordinates of the boundary box as decimal integers    
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:                                                  # The resulting "img" objects are stored inside the vector "all_imgs"
            all_imgs += [img]
          
    return all_imgs, seen_labels

## --------------------------------------------------------------------------------- ##
## ImageReader: Class that contains all the routines necesary for image "manipulation"
## --------------------------------------------------------------------------------- ##

import numpy as np
import cv2
import copy

class ImageReader(object):

    def __init__(self, IMAGE_H, IMAGE_W, norm=None):                                # Method called when an "ImageReader" object is created                             # Routine that executes everytime that the class is called
        '''
        Initialization of parameters IMAGE_H, IMAGE_W and norm

        Parameters
        -----------
        
        IMAGE_H (int)
            The height of the rescaled image. Prefered value = 416

        IMAGE_W (int)
            The width of the rescaled image. Prefered value = 416

        norm (func)
            Normalization function. Prefered func = ImageMatrix/255
        '''

        self.IMAGE_H = IMAGE_H                                                      # The object created will have a field called "IMAGE_H"               
        self.IMAGE_W = IMAGE_W                                                      # The object created will have a field called "IMAGE_W"
        self.norm = norm

    def encode_core(self, image, reorder_rgb=True):                          
        '''
        Resizing of an image to a standard IMAGE_H x IMAGE_W size

        Parameters
        -----------
        
        image (dimensions = image width, image height, 3)
            Pixel values extracted from an image
        
        reorder_rgb (bool)
            If true, reorders the channels of the rescaled image (RGB to BGR)

        Returns
        --------

        image (pixel data)
            Reordered and normalized (if enabled) image data
        '''
        image = cv2.resize(image, (self.IMAGE_H, self.IMAGE_W))                     # Resizing to a IMAGE_H x IMAGE_W size

        if reorder_rgb:
            image = image[:, :, ::-1]                                               # If "reorder_rgb" == True, then the RGB channels of the input matrix are switched to a BGR matrix.
        
        if self.norm is not None:                                                   # If norm is not "None" a normalized version of the image replaces the original data
            image = self.norm(image)

        return(image)

    def fit(self, train_instance):
        '''
        Read the image properties and use its height and width to resize
        the dimensions of the bounding boxes described in the corresponding
        annotation file.
        
        Parameters
        -----------
        
        train_instance (dictionary) 
            Dictionary containing filename, height, width and object. Example:
        
                {'filename': 'JPEGImages/2008_000054.jpg',
                'height':   333,
                'width':    500,
                'object': [{'name': 'bird',
                            'xmax': 318,
                            'xmin': 284,
                            'ymax': 184,
                            'ymin': 100},
                            {'name': 'bird',
                            'xmax': 198,
                            'xmin': 112,
                            'ymax': 209,
                            'ymin': 146}]
                }

        Returns
        --------

        image (pixel image data)
            Resized and "channel reordered" image.

        all_objs (dictionary)
            Scaled "xmin", "xmax", "ymax" and "ymin" values

        '''

        if not isinstance(train_instance, dict):                            # If "train_instance" is not a dictionary
            train_instance = {'filename':train_instance}                    # A new dictionary is created where "train_instance" is the filename
     
        image_name = train_instance['filename']                             # The file name is extracted and saved

        image = cv2.imread(image_name)                                      # The image data is stored in a matrix
        h, w, c = image.shape                                               # Extract the "height", "width" and "color channels"
        if image is None: print('Cannot find ', image_name)                 # If the data extracted is non existent, an error is returned

        image = self.encode_core(image, reorder_rgb=True)                   # The image is resized and its RGB channels are re-ordered

        if "object" in train_instance.keys():                               # If the field "object" exists inside train_instance
     
            all_objs = copy.deepcopy(train_instance['object'])              # A deep copy of the "objects" field is stored    

            for obj in all_objs:                                            # The height and width of the b. boxes present in the image are fixed
                for attr in ['xmin', 'xmax']:                               
                    obj[attr] = int(obj[attr] * float(self.IMAGE_W) / w)    # "xmin/xmax" = "xmin/xmax" * NewWidth / OldWidth
                    obj[attr] = max(min(obj[attr], self.IMAGE_W), 0)        # 

                for attr in ['ymin', 'ymax']:
                    obj[attr] = int(obj[attr] * float(self.IMAGE_H) / h)    # "ymin/ymax" = "ymin/ymax" * NewHeight / OldHeight
                    obj[attr] = max(min(obj[attr], self.IMAGE_H), 0)
        else:
            return image

        return image, all_objs                                                  
    
## --------------------------------------------------------------------------------- ##
## BestAnchorBoxFinder: Given anchor box candidates returns the best match with a established size 
## --------------------------------------------------------------------------------- ##

class BestAnchorBoxFinder(object):

    def __init__(self, ANCHORS):
        '''
        An np.array of even number length containing the properties of different
        candidate anchor boxes (width and height pairs). Stored as an array
        of objects with fields: xmin, ymin, xmax and ymax (Through "BoundBox" class).
        xmin and ymin are always 0, which means that the anchor boxes stored in each
        element of the array have their upper left corner aligned with the origin.  

        Parameters
        -----------

        ANCHORS (list)
            A single dimension np.array of even number length. Example:
        
            ANCHORS = [4,2, ##  width=4, height=2,  flat large anchor box
                       2,4, ##  width=2, height=4,  tall large anchor box
                       1,1] ##  width=1, height=1,  small anchor box
        '''
        self.anchors = [BoundBox(0, 0, ANCHORS[2*i], ANCHORS[2*i+1])       
                        for i in range(int(len(ANCHORS)//2))]
        
    def _interval_overlap(self, interval_a, interval_b):
        '''
        Given the xmin and xmax parameters (Or ymin and ymax) of two different anchor
        boxes, this routine returns the amount of overlap between said anchor boxes in
        an specific axis. If the user gives xmin and xmax values of two anchor boxes, 
        for example, the program will return the overlap in the x axis.

        Parameters
        -----------

        interval_a (1,2)
            Array with the xmin/xmax or ymin/ymax pairs for an anchor box 1

        interval_b (1,2)
            Array with the xmin/xmax or ymin/ymax pairs for an anchor box 2

        Returns
        ---------

        intersect (double)
            Amount of overlap (scalar) in a given direction 

        '''

        x1, x2 = interval_a                             # xmin and xmax (or ymin and ymax) of the anchor box 1 (AB1)
        x3, x4 = interval_b                             # xmin and xmax (or ymin and ymax) of the anchor box 2 (AB2)
        
        if x3 < x1:                                     # If "min" value of AB2 < "min" value of AB1: AB2 is to the left (or above if using "y") of AB1
            if x4 < x1:                                     # If "max" value of AB2 < "min" value of AB2: There is no overlap 
                return 0
            else:                                           # If "max" value of AB2 > "min" value of AB2:
                return min(x2,x4) - x1                      # Overlap = minimum(max AB2, max AB1) - "min" value of AB1

        else:                                           # Else: AB2 is to the right (or below if using "y") of AB1
            if x2 < x3:                                     # If "max" value of AB1 < "min" value of AB2: There is no overlap
                 return 0 
            else:                                           # If "max" value of AB1 > "min" value of AB2: AB2        
                return min(x2,x4) - x3                      # Overlap = minimum(max AB2, max AB1) - "min" value of AB1

    def bbox_iou(self, box1, box2):
        '''
        Calculate the IoU (Intersection Over Union) between two boundary/anchor 
        boxes.

        Parameters
        -----------

        box1 (object)
            BoundBox object corresponding to boundary box 1

        box2 (object)
            BoundBox object corresponding to boundary box 2

        Returns
        ---------

        iou (float)
            Intersection over union between the two input boundary boxes

        '''

        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])        # Intersection on the x axis
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])        # Intersection on the y axis

        intersect = intersect_w * intersect_h                                                       # Area of intersection = Intersection in "x" X Intersection in "y"

        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin                                           # Width and height of boundary boxes 1 and 2
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

        union = w1*h1 + w2*h2 - intersect                                                           # Union = Area of bounding box 1 + Area of bounding box 2 - Area of intersection

        return float(intersect) / union                                                             # Return intersection/union or IoU
    
    def find(self, center_w, center_h):
        '''
        Returns the predefined anchor box that best predicts the dimensions of the
        given bounding box or in other words, returns the anchor box that has the
        highest IoU with the given bounding box (Once the upper left corner of both
        has been aligned with the point 0,0 or the origin).

        Parameters
        -----------

        center_w (int)
            Width of the boundary box given
        
        center_h (int)
            Height of the boundary box given

        Outputs
        --------

        best_anchor (int)
            Index or label of the predefined anchor box with the highest IoU with 
            the given boundary box
        
        max_iou (float)
            Value of the highest IoU

        '''

        best_anchor = -1                                                # The best_anchor starts as a -1 index 
        max_iou     = -1                                                # The max_iou starts as a -1 index
        
        shifted_box = BoundBox(0, 0, center_w, center_h)                # The top left corner of the given boundary box is aligned with the origin of a cell
 
        for i in range(len(self.anchors)):                              # Run through every pre-defined anchor box
            anchor = self.anchors[i]                                    # Stores in "anchor" each of the predefined anchor boxes.
            iou    = self.bbox_iou(shifted_box, anchor)                 # IoU between the given boundary box and the predefined anchor box.
            
            if max_iou < iou:                                           # If the "max_iou" until now is smaller than the current iou
                best_anchor = i                                         # The ith anchor is the best anchor   
                max_iou     = iou                                       # And the "max_iou" is the current iou

        return(best_anchor,max_iou)    

## --------------------------------------------------------------------------------- ##
## BoundBox: Class that contains the properties of a specific boundary box
## --------------------------------------------------------------------------------- ##

class BoundBox:

    def __init__(self, xmin, ymin, xmax, ymax, confidence=None, classes=None):
        '''
        Initialization of the parameters xmin, ymin, xmax and ymax.
        The arrays "confidence" and "classes" will also be stored in the object.

        Parameters
        -----------

        xmin
            X coordinate of the top left corner of the boundary box
        
        xmax
            Y coordinate of the top left corner of the boundary box
        
        ymin
            X coordinate of the bottom right corner of the boundary box
        
        ymax
            Y coordinate of the bottom right corner of the boundary box

        '''
        
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax

        self.confidence      = confidence                               # Used in inference probability
        self.set_class(classes)                                         # Class probaiblities [c1, c2, .. cNclass]
      
    def set_class(self,classes):
        '''
        Saves an array of all the class probabilities detected in a boundary
        box, then extracts the detection label by obtaining the max class
        probability.
        '''

        self.classes = classes                                          # The object stores an array of class probabilities
        self.label   = np.argmax(self.classes)                          # The label for the boundary box will be the class with the highest probability
        
    def get_label(self):
        '''
        Returns the index corresponding to the highest class score.
        This doesn't return the actual score.
        '''

        return(self.label)
    
    def get_score(self):
        '''
        Returns the actual score of the class with the highest probability of
        detection.
        '''

        return(self.classes[self.label])
            
def rescale_centerxy(obj,config):
    '''
    obj:     dictionary containing xmin, xmax, ymin, ymax
    config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
    '''
    center_x = .5*(obj['xmin'] + obj['xmax'])
    center_x = center_x / (float(config['IMAGE_W']) / config['GRID_W'])
    center_y = .5*(obj['ymin'] + obj['ymax'])
    center_y = center_y / (float(config['IMAGE_H']) / config['GRID_H'])
    return(center_x,center_y)

def rescale_cebterwh(obj,config):
    '''
    obj:     dictionary containing xmin, xmax, ymin, ymax
    config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
    '''    
    # unit: grid cell
    center_w = (obj['xmax'] - obj['xmin']) / (float(config['IMAGE_W']) / config['GRID_W']) 
    # unit: grid cell
    center_h = (obj['ymax'] - obj['ymin']) / (float(config['IMAGE_H']) / config['GRID_H']) 
    return(center_w,center_h)

from keras.utils import Sequence

class SimpleBatchGenerator(Sequence):
    def __init__(self, images, config, norm=None, shuffle=True):
        '''
        config : dictionary containing necessary hyper parameters for traning. e.g., 
            {
            'IMAGE_H'         : 416, 
            'IMAGE_W'         : 416,
            'GRID_H'          : 13,  
            'GRID_W'          : 13,
            'LABELS'          : ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
                                  'bus',        'car',      'cat',  'chair',     'cow',
                                  'diningtable','dog',    'horse',  'motorbike', 'person',
                                  'pottedplant','sheep',  'sofa',   'train',   'tvmonitor'],
            'ANCHORS'         : array([ 1.07709888,   1.78171903,  
                                        2.71054693,   5.12469308, 
                                        10.47181473, 10.09646365,  
                                        5.48531347,   8.11011331]),
            'BATCH_SIZE'      : 16,
            'TRUE_BOX_BUFFER' : 50,
            }
        
        '''
        self.config = config
        self.config["BOX"] = int(len(self.config['ANCHORS'])/2)
        self.config["CLASS"] = len(self.config['LABELS'])
        self.images = images
        self.bestAnchorBoxFinder = BestAnchorBoxFinder(config['ANCHORS'])
        self.imageReader = ImageReader(config['IMAGE_H'],config['IMAGE_W'],norm=norm)
        self.shuffle = shuffle
        if self.shuffle: 
            np.random.shuffle(self.images)
            
    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))  
    
    def __getitem__(self, idx):
        '''
        == input == 
        
        idx : non-negative integer value e.g., 0
        
        == output ==
        
        x_batch: The numpy array of shape  (BATCH_SIZE, IMAGE_H, IMAGE_W, N channels).
            
            x_batch[iframe,:,:,:] contains a iframeth frame of size  (IMAGE_H,IMAGE_W).
            
        y_batch:

            The numpy array of shape  (BATCH_SIZE, GRID_H, GRID_W, BOX, 4 + 1 + N classes). 
            BOX = The number of anchor boxes.

            y_batch[iframe,igrid_h,igrid_w,ianchor,:4] contains (center_x,center_y,center_w,center_h) 
            of ianchorth anchor at  grid cell=(igrid_h,igrid_w) if the object exists in 
            this (grid cell, anchor) pair, else they simply contain 0.

            y_batch[iframe,igrid_h,igrid_w,ianchor,4] contains 1 if the object exists in this 
            (grid cell, anchor) pair, else it contains 0.

            y_batch[iframe,igrid_h,igrid_w,ianchor,5 + iclass] contains 1 if the iclass^th 
            class object exists in this (grid cell, anchor) pair, else it contains 0.


        b_batch:

            The numpy array of shape (BATCH_SIZE, 1, 1, 1, TRUE_BOX_BUFFER, 4).

            b_batch[iframe,1,1,1,ibuffer,ianchor,:] contains ibufferth object's 
            (center_x,center_y,center_w,center_h) in iframeth frame.

            If ibuffer > N objects in iframeth frame, then the values are simply 0.

            TRUE_BOX_BUFFER has to be some large number, so that the frame with the 
            biggest number of objects can also record all objects.

            The order of the objects do not matter.

            This is just a hack to easily calculate loss. 
        
        '''
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0
        
        ## prepare empty storage space: this will be output
        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        b_batch = np.zeros((r_bound - l_bound, 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))                # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.imageReader.fit(train_instance)
            
            # construct output from object's x, y, w, h
            true_box_index = 0
            
            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                    center_x, center_y = rescale_centerxy(obj,self.config)
                    
                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx  = self.config['LABELS'].index(obj['name'])
                        center_w, center_h = rescale_cebterwh(obj,self.config)
                        box = [center_x, center_y, center_w, center_h]
                        best_anchor,max_iou = self.bestAnchorBoxFinder.find(center_w, center_h)
                                
                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        # it could happen that the same grid cell contain 2 similar shape objects
                        # as a result the same anchor box is selected as the best anchor box by the multiple objects
                        # in such ase, the object is over written
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box # center_x, center_y, w, h
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4  ] = 1. # ground truth confidence is 1
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5+obj_indx] = 1 # class probability of the object
                        
                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box
                        
                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
                            
            x_batch[instance_count] = img
            # increase instance counter in current batch
            instance_count += 1  
        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: 
            np.random.shuffle(self.images)
## ==============================================================            
## Part 3 Object Detection with Yolo using VOC 2014 data - model
## ==============================================================

from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
import keras.backend as K
import tensorflow as tf

# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)

def ConvBatchLReLu(x,filters,kernel_size,index,trainable):
    # when strides = None, strides = pool_size.
    x = Conv2D(filters, kernel_size, strides=(1,1), 
               padding='same', name='conv_{}'.format(index), 
               use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_{}'.format(index), trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return(x)
def ConvBatchLReLu_loop(x,index,convstack,trainable):
    for para in convstack:
        x = ConvBatchLReLu(x,para["filters"],para["kernel_size"],index,trainable)
        index += 1
    return(x)
def define_YOLOv2(IMAGE_H,IMAGE_W,GRID_H,GRID_W,TRUE_BOX_BUFFER,BOX,CLASS, trainable=False):
    convstack3to5  = [{"filters":128, "kernel_size":(3,3)},  # 3
                      {"filters":64,  "kernel_size":(1,1)},  # 4
                      {"filters":128, "kernel_size":(3,3)}]  # 5
                    
    convstack6to8  = [{"filters":256, "kernel_size":(3,3)},  # 6
                      {"filters":128, "kernel_size":(1,1)},  # 7
                      {"filters":256, "kernel_size":(3,3)}]  # 8
    
    convstack9to13 = [{"filters":512, "kernel_size":(3,3)},  # 9
                      {"filters":256, "kernel_size":(1,1)},  # 10
                      {"filters":512, "kernel_size":(3,3)},  # 11
                      {"filters":256, "kernel_size":(1,1)},  # 12
                      {"filters":512, "kernel_size":(3,3)}]  # 13
        
    convstack14to20 = [{"filters":1024, "kernel_size":(3,3)}, # 14 
                       {"filters":512,  "kernel_size":(1,1)}, # 15
                       {"filters":1024, "kernel_size":(3,3)}, # 16
                       {"filters":512,  "kernel_size":(1,1)}, # 17
                       {"filters":1024, "kernel_size":(3,3)}, # 18
                       {"filters":1024, "kernel_size":(3,3)}, # 19
                       {"filters":1024, "kernel_size":(3,3)}] # 20
    
    input_image = Input(shape=(IMAGE_H, IMAGE_W, 3),name="input_image")
    true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4),name="input_hack")    
    # Layer 1
    x = ConvBatchLReLu(input_image,filters=32,kernel_size=(3,3),index=1,trainable=trainable)
    
    x = MaxPooling2D(pool_size=(2, 2),name="maxpool1_416to208")(x)
    # Layer 2
    x = ConvBatchLReLu(x,filters=64,kernel_size=(3,3),index=2,trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2),name="maxpool1_208to104")(x)
    
    # Layer 3 - 5
    x = ConvBatchLReLu_loop(x,3,convstack3to5,trainable)
    x = MaxPooling2D(pool_size=(2, 2),name="maxpool1_104to52")(x)
    
    # Layer 6 - 8 
    x = ConvBatchLReLu_loop(x,6,convstack6to8,trainable)
    x = MaxPooling2D(pool_size=(2, 2),name="maxpool1_52to26")(x) 

    # Layer 9 - 13
    x = ConvBatchLReLu_loop(x,9,convstack9to13,trainable)
        
    skip_connection = x
    x = MaxPooling2D(pool_size=(2, 2),name="maxpool1_26to13")(x)
    
    # Layer 14 - 20
    x = ConvBatchLReLu_loop(x,14,convstack14to20,trainable)

    # Layer 21
    skip_connection = ConvBatchLReLu(skip_connection,filters=64,
                                     kernel_size=(1,1),index=21,trainable=trainable)
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = ConvBatchLReLu(x,filters=1024,kernel_size=(3,3),index=22,trainable=trainable)

    # Layer 23
    x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
    output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS),name="final_output")(x)

    # small hack to allow true_boxes to be registered when Keras build the model 
    # for more information: https://github.com/fchollet/keras/issues/2790
    output = Lambda(lambda args: args[0],name="hack_layer")([output, true_boxes])

    model = Model([input_image, true_boxes], output)
    return(model, true_boxes)

class WeightReader:
    # code from https://github.com/experiencor/keras-yolo2/blob/master/Yolo%20Step-by-Step.ipynb
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4
        
def set_pretrained_weight(model,nb_conv, path_to_weight):
    weight_reader = WeightReader(path_to_weight)
    weight_reader.reset()
    for i in range(1, nb_conv+1):
        conv_layer = model.get_layer('conv_' + str(i)) ## convolusional layer

        if i < nb_conv:
            norm_layer = model.get_layer('norm_' + str(i)) ## batch normalization layer

            size = np.prod(norm_layer.get_weights()[0].shape)

            beta  = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean  = weight_reader.read_bytes(size)
            var   = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])       

        if len(conv_layer.get_weights()) > 1: ## with bias
            bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel, bias])
        else: ## without bias
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel])
    return(model)
        

def initialize_weight(layer,sd):
    weights = layer.get_weights()
    new_kernel = np.random.normal(size=weights[0].shape, scale=sd)
    new_bias   = np.random.normal(size=weights[1].shape, scale=sd)
    layer.set_weights([new_kernel, new_bias])
    
    
## ==============================================================            
## Part 4 Object Detection with Yolo using VOC 2014 data - loss
## ==============================================================

def get_cell_grid(GRID_W,GRID_H,BATCH_SIZE,BOX): 
    '''
    Helper function to assure that the bounding box x and y are in the grid cell scale
    == output == 
    for any i=0,1..,batch size - 1
    output[i,5,3,:,:] = array([[3., 5.],
                               [3., 5.],
                               [3., 5.]], dtype=float32)
    '''
    ## cell_x.shape = (1, 13, 13, 1, 1)
    ## cell_x[:,i,j,:] = [[[j]]]
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
    ## cell_y.shape = (1, 13, 13, 1, 1)
    ## cell_y[:,i,j,:] = [[[i]]]
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))
    ## cell_gird.shape = (16, 13, 13, 5, 2)
    ## for any n, k, i, j
    ##    cell_grid[n, i, j, anchor, k] = j when k = 0
    ## for any n, k, i, j
    ##    cell_grid[n, i, j, anchor, k] = i when k = 1    
    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [BATCH_SIZE, 1, 1, BOX, 1])
    return(cell_grid) 

def adjust_scale_prediction(y_pred, cell_grid, ANCHORS):    
    """
        Adjust prediction
        
        == input ==
        
        y_pred : takes any real values
                 tensor of shape = (N batch, NGrid h, NGrid w, NAnchor, 4 + 1 + N class)
        
        ANCHORS : list containing width and height specializaiton of anchor box
        == output ==
        
        pred_box_xy : shape = (N batch, N grid x, N grid y, N anchor, 2), contianing [center_y, center_x] rangining [0,0]x[grid_H-1,grid_W-1]
          pred_box_xy[irow,igrid_h,igrid_w,ianchor,0] =  center_x
          pred_box_xy[irow,igrid_h,igrid_w,ianchor,1] =  center_1
          
          calculation process:
          tf.sigmoid(y_pred[...,:2]) : takes values between 0 and 1
          tf.sigmoid(y_pred[...,:2]) + cell_grid : takes values between 0 and grid_W - 1 for x coordinate 
                                                   takes values between 0 and grid_H - 1 for y coordinate 
                                                   
        pred_Box_wh : shape = (N batch, N grid h, N grid w, N anchor, 2), 
                        containing width and height, rangining [0,0]x[grid_H-1,grid_W-1]
        
        pred_box_conf : shape = (N batch, N grid h, N grid w, N anchor, 1), containing confidence to range between 0 and 1
        
        pred_box_class : shape = (N batch, N grid h, N grid w, N anchor, N class), containing 
    """
    BOX = int(len(ANCHORS)/2)
    ## cell_grid is of the shape of 
    
    ### adjust x and y  
    # the bounding box bx and by are rescaled to range between 0 and 1 for given gird.
    # Since there are BOX x BOX grids, we rescale each bx and by to range between 0 to BOX + 1
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid # bx, by
    
    ### adjust w and h
    # exp to make width and height positive
    # rescale each grid to make some anchor "good" at representing certain shape of bounding box 
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS,[1,1,1,BOX,2]) # bw, bh

    ### adjust confidence 
    pred_box_conf = tf.sigmoid(y_pred[..., 4])# prob bb

    ### adjust class probabilities 
    pred_box_class = y_pred[..., 5:] # prC1, prC2, ..., prC20
    
    return(pred_box_xy,pred_box_wh,pred_box_conf,pred_box_class)

def extract_ground_truth(y_true):    
    true_box_xy    = y_true[..., 0:2] # bounding box x, y coordinate in grid cell scale 
    true_box_wh    = y_true[..., 2:4] # number of cells accross, horizontally and vertically
    true_box_conf  = y_true[...,4]    # confidence 
    true_box_class = tf.argmax(y_true[..., 5:], -1)
    return(true_box_xy, true_box_wh, true_box_conf, true_box_class)

def calc_loss_xywh(true_box_conf,
                   COORD_SCALE,
                   true_box_xy, pred_box_xy,true_box_wh,pred_box_wh):  
    '''
    coord_mask:      np.array of shape (Nbatch, Ngrid h, N grid w, N anchor, 1)
                     lambda_{coord} L_{i,j}^{obj}     
                         
    '''
    
    # lambda_{coord} L_{i,j}^{obj} 
    # np.array of shape (Nbatch, Ngrid h, N grid w, N anchor, 1)
    coord_mask  = tf.expand_dims(true_box_conf, axis=-1) * COORD_SCALE 
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    loss_xy      = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh      = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    return(loss_xy + loss_wh, coord_mask)
def calc_loss_class(true_box_conf,CLASS_SCALE, true_box_class,pred_box_class):
    '''
    == input ==    
    true_box_conf  : tensor of shape (N batch, N grid h, N grid w, N anchor)
    true_box_class : tensor of shape (N batch, N grid h, N grid w, N anchor), containing class index
    pred_box_class : tensor of shape (N batch, N grid h, N grid w, N anchor, N class)
    CLASS_SCALE    : 1.0
    
    == output ==  
    class_mask
    if object exists in this (grid_cell, anchor) pair and the class object receive nonzero weight
        class_mask[iframe,igridy,igridx,ianchor] = 1 
    else: 
        0 
    '''   
    class_mask   = true_box_conf  * CLASS_SCALE ## L_{i,j}^obj * lambda_class
    
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
    loss_class   = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = true_box_class, 
                                                                  logits = pred_box_class)
    loss_class   = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)   
    return(loss_class)



def get_intersect_area(true_xy,true_wh,
                       pred_xy,pred_wh):
    '''
    == INPUT ==
    true_xy,pred_xy, true_wh and pred_wh must have the same shape length

    p1 : pred_mins = (px1,py1)
    p2 : pred_maxs = (px2,py2)
    t1 : true_mins = (tx1,ty1) 
    t2 : true_maxs = (tx2,ty2) 
                 p1______________________ 
                 |      t1___________   |
                 |       |           |  |
                 |_______|___________|__|p2 
                         |           |rmax
                         |___________|
                                      t2
    intersect_mins : rmin = t1  = (tx1,ty1)
    intersect_maxs : rmax = (rmaxx,rmaxy)
    intersect_wh   : (rmaxx - tx1, rmaxy - ty1)
        
    '''
    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half
    
    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half    
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)    
    return(iou_scores)

def calc_IOU_pred_true_assigned(true_box_conf,
                                true_box_xy, true_box_wh,
                                pred_box_xy,  pred_box_wh):
    ''' 
    == input ==
    
    true_box_conf : tensor of shape (N batch, N grid h, N grid w, N anchor )
    true_box_xy   : tensor of shape (N batch, N grid h, N grid w, N anchor , 2)
    true_box_wh   : tensor of shape (N batch, N grid h, N grid w, N anchor , 2)
    pred_box_xy   : tensor of shape (N batch, N grid h, N grid w, N anchor , 2)
    pred_box_wh   : tensor of shape (N batch, N grid h, N grid w, N anchor , 2)
        
    == output ==
    
    true_box_conf : tensor of shape (N batch, N grid h, N grid w, N anchor)
    
    true_box_conf value depends on the predicted values 
    true_box_conf = IOU_{true,pred} if objecte exist in this anchor else 0
    '''
    iou_scores        =  get_intersect_area(true_box_xy,true_box_wh,
                                            pred_box_xy,pred_box_wh)
    true_box_conf_IOU = iou_scores * true_box_conf
    return(true_box_conf_IOU)


def calc_IOU_pred_true_best(pred_box_xy,pred_box_wh,true_boxes):   
    '''
    == input ==
    pred_box_xy : tensor of shape (N batch, N grid h, N grid w, N anchor, 2)
    pred_box_wh : tensor of shape (N batch, N grid h, N grid w, N anchor, 2)
    true_boxes  : tensor of shape (N batch, N grid h, N grid w, N anchor, 2)
    
    == output == 
    
    best_ious
    
    for each iframe,
        best_ious[iframe,igridy,igridx,ianchor] contains
        
        the IOU of the object that is most likely included (or best fitted) 
        within the bounded box recorded in (grid_cell, anchor) pair
        
        NOTE: a same object may be contained in multiple (grid_cell, anchor) pair
              from best_ious, you cannot tell how may actual objects are captured as the "best" object
    '''
    true_xy = true_boxes[..., 0:2]           # (N batch, 1, 1, 1, TRUE_BOX_BUFFER, 2)
    true_wh = true_boxes[..., 2:4]           # (N batch, 1, 1, 1, TRUE_BOX_BUFFER, 2)
    
    pred_xy = tf.expand_dims(pred_box_xy, 4) # (N batch, N grid_h, N grid_w, N anchor, 1, 2)
    pred_wh = tf.expand_dims(pred_box_wh, 4) # (N batch, N grid_h, N grid_w, N anchor, 1, 2)
    
    iou_scores  =  get_intersect_area(true_xy,
                                      true_wh,
                                      pred_xy,
                                      pred_wh) # (N batch, N grid_h, N grid_w, N anchor, 50)   

    best_ious = tf.reduce_max(iou_scores, axis=4) # (N batch, N grid_h, N grid_w, N anchor)
    return(best_ious)
def get_conf_mask(best_ious, true_box_conf, true_box_conf_IOU,LAMBDA_NO_OBJECT, LAMBDA_OBJECT):    
    '''
    == input == 
    
    best_ious           : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
    true_box_conf       : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
    true_box_conf_IOU   : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
    LAMBDA_NO_OBJECT    : 1.0
    LAMBDA_OBJECT       : 5.0
    
    == output ==
    conf_mask : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
    
    conf_mask[iframe, igridy, igridx, ianchor] = 0
               when there is no object assigned in (grid cell, anchor) pair and the region seems useless i.e. 
               y_true[iframe,igridx,igridy,4] = 0 "and" the predicted region has no object that has IoU > 0.6
               
    conf_mask[iframe, igridy, igridx, ianchor] =  NO_OBJECT_SCALE
               when there is no object assigned in (grid cell, anchor) pair but region seems to include some object
               y_true[iframe,igridx,igridy,4] = 0 "and" the predicted region has some object that has IoU > 0.6
               
    conf_mask[iframe, igridy, igridx, ianchor] =  OBJECT_SCALE
              when there is an object in (grid cell, anchor) pair        
    '''

    conf_mask = tf.to_float(best_ious < 0.6) * (1 - true_box_conf) * LAMBDA_NO_OBJECT
    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + true_box_conf_IOU * LAMBDA_OBJECT
    return(conf_mask)

def calc_loss_conf(conf_mask,true_box_conf_IOU, pred_box_conf):  
    '''
    == input ==
    
    conf_mask         : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
    true_box_conf_IOU : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
    pred_box_conf     : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
    '''
    # the number of (grid cell, anchor) pair that has an assigned object or
    # that has no assigned object but some objects may be in bounding box.
    # N conf
    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
    loss_conf    = tf.reduce_sum(tf.square(true_box_conf_IOU-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
    return(loss_conf)

def custom_loss_core(y_true,
                     y_pred,
                     true_boxes,
                     GRID_W,
                     GRID_H,
                     BATCH_SIZE,
                     ANCHORS,
                     LAMBDA_COORD,
                     LAMBDA_CLASS,
                     LAMBDA_NO_OBJECT, 
                     LAMBDA_OBJECT):
    '''
    y_true : (N batch, N grid h, N grid w, N anchor, 4 + 1 + N classes)
    y_true[irow, i_gridh, i_gridw, i_anchor, :4] = center_x, center_y, w, h
    
        center_x : The x coordinate center of the bounding box.
                   Rescaled to range between 0 and N gird  w (e.g., ranging between [0,13)
        center_y : The y coordinate center of the bounding box.
                   Rescaled to range between 0 and N gird  h (e.g., ranging between [0,13)
        w        : The width of the bounding box.
                   Rescaled to range between 0 and N gird  w (e.g., ranging between [0,13)
        h        : The height of the bounding box.
                   Rescaled to range between 0 and N gird  h (e.g., ranging between [0,13)
                   
    y_true[irow, i_gridh, i_gridw, i_anchor, 4] = ground truth confidence
        
        ground truth confidence is 1 if object exists in this (anchor box, gird cell) pair
    
    y_true[irow, i_gridh, i_gridw, i_anchor, 5 + iclass] = 1 if the object is in category <iclass> else 0
    
    =====================================================
    tensor that connect to the YOLO model's hack input 
    =====================================================    
    
    true_boxes    
    
    =========================================
    training parameters specification example 
    =========================================
    GRID_W             = 13
    GRID_H             = 13
    BATCH_SIZE         = 34
    ANCHORS = np.array([1.07709888,  1.78171903,  # anchor box 1, width , height
                        2.71054693,  5.12469308,  # anchor box 2, width,  height
                       10.47181473, 10.09646365,  # anchor box 3, width,  height
                        5.48531347,  8.11011331]) # anchor box 4, width,  height
    LAMBDA_NO_OBJECT = 1.0
    LAMBDA_OBJECT    = 5.0
    LAMBDA_COORD     = 1.0
    LAMBDA_CLASS     = 1.0
    ''' 
    BOX = int(len(ANCHORS)/2)    
    # Step 1: Adjust prediction output
    cell_grid   = get_cell_grid(GRID_W,GRID_H,BATCH_SIZE,BOX)
    pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = adjust_scale_prediction(y_pred,cell_grid,ANCHORS)
    # Step 2: Extract ground truth output
    true_box_xy, true_box_wh, true_box_conf, true_box_class = extract_ground_truth(y_true)
    # Step 3: Calculate loss for the bounding box parameters
    loss_xywh, coord_mask = calc_loss_xywh(true_box_conf,LAMBDA_COORD,
                                           true_box_xy, pred_box_xy,true_box_wh,pred_box_wh)
    # Step 4: Calculate loss for the class probabilities
    loss_class  = calc_loss_class(true_box_conf,LAMBDA_CLASS,
                                   true_box_class,pred_box_class)
    # Step 5: For each (grid cell, anchor) pair, 
    #         calculate the IoU between predicted and ground truth bounding box
    true_box_conf_IOU = calc_IOU_pred_true_assigned(true_box_conf,
                                                    true_box_xy, true_box_wh,
                                                    pred_box_xy, pred_box_wh)
    # Step 6: For each predicted bounded box from (grid cell, anchor box), 
    #         calculate the best IOU, regardless of the ground truth anchor box that each object gets assigned.
    best_ious = calc_IOU_pred_true_best(pred_box_xy,pred_box_wh,true_boxes)
    # Step 7: For each grid cell, calculate the L_{i,j}^{noobj}
    conf_mask = get_conf_mask(best_ious, true_box_conf, true_box_conf_IOU,LAMBDA_NO_OBJECT, LAMBDA_OBJECT)
    # Step 8: Calculate loss for the confidence
    loss_conf = calc_loss_conf(conf_mask,true_box_conf_IOU, pred_box_conf)
    
    loss = loss_xywh + loss_conf + loss_class
    return(loss)

def custom_loss(y_true, y_pred):
    loss = custom_loss_core(y_true,
                     y_pred,
                     true_boxes,
                     GRID_W,
                     GRID_H,
                     BATCH_SIZE,
                     ANCHORS,
                     LAMBDA_COORD,
                     LAMBDA_CLASS,
                     LAMBDA_NO_OBJECT, 
                     LAMBDA_OBJECT)

    
    return loss


# ========================================================================== ##
# Part 6 Object Detection with Yolo using VOC 2012 data - inference on image
# ========================================================================== ##

class OutputRescaler(object):
    def __init__(self,ANCHORS):
        self.ANCHORS = ANCHORS

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
    def _softmax(self, x, axis=-1, t=-100.):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x/np.min(x)*t

        e_x = np.exp(x)
        return e_x / e_x.sum(axis, keepdims=True)
    def get_shifting_matrix(self,netout):
        
        GRID_H, GRID_W, BOX = netout.shape[:3]
        no = netout[...,0]
        
        ANCHORSw = self.ANCHORS[::2]
        ANCHORSh = self.ANCHORS[1::2]
       
        mat_GRID_W = np.zeros_like(no)
        for igrid_w in range(GRID_W):
            mat_GRID_W[:,igrid_w,:] = igrid_w

        mat_GRID_H = np.zeros_like(no)
        for igrid_h in range(GRID_H):
            mat_GRID_H[igrid_h,:,:] = igrid_h

        mat_ANCHOR_W = np.zeros_like(no)
        for ianchor in range(BOX):    
            mat_ANCHOR_W[:,:,ianchor] = ANCHORSw[ianchor]

        mat_ANCHOR_H = np.zeros_like(no) 
        for ianchor in range(BOX):    
            mat_ANCHOR_H[:,:,ianchor] = ANCHORSh[ianchor]
        return(mat_GRID_W,mat_GRID_H,mat_ANCHOR_W,mat_ANCHOR_H)

    def fit(self, netout):    
        '''
        netout  : np.array of shape (N grid h, N grid w, N anchor, 4 + 1 + N class)
        
        a single image output of model.predict()
        '''
        GRID_H, GRID_W, BOX = netout.shape[:3]
        
        (mat_GRID_W,
         mat_GRID_H,
         mat_ANCHOR_W,
         mat_ANCHOR_H) = self.get_shifting_matrix(netout)


        # bounding box parameters
        netout[..., 0]   = (self._sigmoid(netout[..., 0]) + mat_GRID_W)/GRID_W # x      unit: range between 0 and 1
        netout[..., 1]   = (self._sigmoid(netout[..., 1]) + mat_GRID_H)/GRID_H # y      unit: range between 0 and 1
        netout[..., 2]   = (np.exp(netout[..., 2]) * mat_ANCHOR_W)/GRID_W      # width  unit: range between 0 and 1
        netout[..., 3]   = (np.exp(netout[..., 3]) * mat_ANCHOR_H)/GRID_H      # height unit: range between 0 and 1
        # rescale the confidence to range 0 and 1 
        netout[..., 4]   = self._sigmoid(netout[..., 4])
        expand_conf      = np.expand_dims(netout[...,4],-1) # (N grid h , N grid w, N anchor , 1)
        # rescale the class probability to range between 0 and 1
        # Pr(object class = k) = Pr(object exists) * Pr(object class = k |object exists)
        #                      = Conf * P^c
        netout[..., 5:]  = expand_conf * self._softmax(netout[..., 5:])
        # ignore the class probability if it is less than obj_threshold 
    
        return(netout)
    
    
def find_high_class_probability_bbox(netout_scale, obj_threshold):
    '''
    == Input == 
    netout : y_pred[i] np.array of shape (GRID_H, GRID_W, BOX, 4 + 1 + N class)
    
             x, w must be a unit of image width
             y, h must be a unit of image height
             c must be in between 0 and 1
             p^c must be in between 0 and 1
    == Output ==
    
    boxes  : list containing bounding box with Pr(object is in class C) > 0 for at least in one class C 
    
             
    '''
    GRID_H, GRID_W, BOX = netout_scale.shape[:3]
    
    boxes = []
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                # from 4th element onwards are confidence and class classes
                classes = netout_scale[row,col,b,5:]
                
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout_scale[row,col,b,:4]
                    confidence = netout_scale[row,col,b,4]
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    if box.get_score() > obj_threshold:
                        boxes.append(box)
    return(boxes)

import cv2, copy
import seaborn as sns
def draw_boxes(_image, boxes, labels, obj_baseline=0.05,verbose=False):
    '''
    image : np.array of shape (N height, N width, 3)
    '''
    def adjust_minmax(c,_max):
        if c < 0:
            c = 0   
        if c > _max:
            c = _max
        return c
    
    image = copy.deepcopy(_image)
    image_h, image_w, _ = image.shape
    score_rescaled  = np.array([box.get_score() for box in boxes])
    score_rescaled /= obj_baseline
    color_rect,color_text = sns.color_palette("husl", 2)
    for sr, box in zip(score_rescaled,boxes):
        xmin = adjust_minmax(int(box.xmin*image_w),image_w)
        ymin = adjust_minmax(int(box.ymin*image_h),image_h)
        xmax = adjust_minmax(int(box.xmax*image_w),image_w)
        ymax = adjust_minmax(int(box.ymax*image_h),image_h)
 
        
        text = "{:10} {:4.3f}".format(labels[box.label], box.get_score())
        if verbose:
            print("{} xmin={:4.0f},ymin={:4.0f},xmax={:4.0f},ymax={:4.0f}".format(text,xmin,ymin,xmax,ymax,text))
        cv2.rectangle(image, 
                      pt1       = (xmin,ymin), 
                      pt2       = (xmax,ymax), 
                      color     = color_rect, 
                      thickness = sr)
        cv2.putText(img       = image, 
                    text      = text, 
                    org       = (xmin+ 13, ymin + 13),
                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1e-3 * image_h,
                    color     = color_text,
                    thickness = 1)
        
    return image



def nonmax_suppression(boxes,iou_threshold,obj_threshold):
    '''
    boxes : list containing "good" BoundBox of a frame
            [BoundBox(),BoundBox(),...]
    '''
    bestAnchorBoxFinder    = BestAnchorBoxFinder([])
    
    CLASS    = len(boxes[0].classes)
    index_boxes = []   
    # suppress non-maximal boxes
    for c in range(CLASS):
        # extract class probabilities of the c^th class from multiple bbox
        class_probability_from_bbxs = [box.classes[c] for box in boxes]

        #sorted_indices[i] contains the i^th largest class probabilities
        sorted_indices = list(reversed(np.argsort( class_probability_from_bbxs)))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            # if class probability is zero then ignore
            if boxes[index_i].classes[c] == 0:  
                continue
            else:
                index_boxes.append(index_i)
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    # check if the selected i^th bounding box has high IOU with any of the remaining bbox
                    # if so, the remaining bbox' class probabilities are set to 0.
                    bbox_iou = bestAnchorBoxFinder.bbox_iou(boxes[index_i], boxes[index_j])
                    if bbox_iou >= iou_threshold:
                        classes = boxes[index_j].classes
                        classes[c] = 0
                        boxes[index_j].set_class(classes)
                        
    newboxes = [ boxes[i] for i in index_boxes if boxes[i].get_score() > obj_threshold ]                
    
    return newboxes  
  
## =========================== ##    
## Load Pre-trained weights     
## =========================== ##    

class PreTrainedYOLODetector(object):
    def __init__(self):
        self.LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
                       'bus',        'car',      'cat',  'chair',     'cow',
                       'diningtable','dog',    'horse',  'motorbike', 'person',
                       'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']
        self.ANCHORS = np.array([1.07709888,  1.78171903,  # anchor box 1, width , height
                                 2.71054693,  5.12469308,  # anchor box 2, width,  height
                                10.47181473, 10.09646365,  # anchor box 3, width,  height
                                 5.48531347,  8.11011331]) # anchor box 4, width,  height
        self.BOX                    = int(len(self.ANCHORS)/2)
        self.TRUE_BOX_BUFFER        = 50
        self.IMAGE_H, self.IMAGE_W  = 416, 416
        self.GRID_H,  self.GRID_W   = 13 , 13
        self.CLASS                  = len(self.LABELS)
        self.outputRescaler         = OutputRescaler(ANCHORS = self.ANCHORS)
        self.imageReader            = ImageReader(self.IMAGE_H,
                                                  self.IMAGE_W, 
                                                  norm = lambda image : image / 255.)
    def load(self,path_to_weights):
        model, _          = define_YOLOv2(self.IMAGE_H,
                                          self.IMAGE_W,
                                          self.GRID_H,
                                          self.GRID_W,
                                          self.TRUE_BOX_BUFFER,
                                          self.BOX,
                                          self.CLASS, 
                                  trainable = False)
        self.model = model.load_weights(path_to_weights)
        print("Pretrained weights are loaded")
    def predict(self,X):
        if len(X.shape) == 3:
            X = X.reshape(1,X.shape[0],X.shape[1],X.shape[2])
            dummy_array  = np.zeros((X.shape[0],1,1,1,self.TRUE_BOX_BUFFER,4))
        y_pred   =  self.model.predict([X,dummy_array])
        return(y_pred)