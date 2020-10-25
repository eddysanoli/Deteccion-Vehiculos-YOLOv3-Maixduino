import os, cv2
import matplotlib.pyplot as plt
import numpy as np 
from backend import *
import kmeans

Labels = ["car"]                                            # Vector that contains all the class names to detect
raw_images = "raw_dataset/images/"                          # Directory containing training images
raw_annots = "raw_dataset/annotations/"                     # Directory containing training annotations
train_images = "train_dataset/images/"                      # Directory containing training images
train_annots = "train_dataset/annotations/"                 # Directory containing training annotations

# EXTRACTING A SAMPLE ***************************************************
print("\nTask: Sample Extraction")
# Extract a sample size of images from the raw dataset
extract_sample(raw_annots, train_annots, raw_images, train_images, 1000)


# ANNOTATION PARSING ***************************************************
print("\nTask: Annotation Parsing")
# Parse annotations, count the number of classes present in the dataset
# and count the number of training samples

train_images, seen_train_labels = parse_annotations(train_annots, train_images, labels=Labels)

figure, axes = plt.subplots()                                                                   # A plot is created
y_pos = np.arange(len(seen_train_labels))                                                       # Number of ticks in the "y" axis, one for each class
axes.barh(y_pos, list(seen_train_labels.values()))                                              # The length of each bar is set to the amount of objects per class
axes.set_yticks(y_pos)                                                                          # Creation of the ticks in the "y" axis
axes.set_yticklabels(list(seen_train_labels.keys()))                                            # Each tick in the "y" axis takes the name of one class
axes.set_title("{} class(es) in {} images".format(len(seen_train_labels), len(train_images)))   # Plot title.
plt.show() 


# K-MEANS CLUSTERING ***************************************************
print("\nTask: K-Means Clustering")
# Generate appropriate anchor box dimensions based on the available data.
# For this process, the YOLO9000 paper suggests the use of the K-Means clustering
# algorithm. The YOLO algorithm can use random anchor box sizes, however, a good
# set of prior anchor boxes can lead to better predictions.

BBoxDimensions = []

for annot in train_images:
    im_w = float(annot['width'])                            # Width of the original image
    im_h = float(annot['height'])                           # Height of the original image

    for obj in annot['object']:                             # Parses through every bounding box contained inside the "object" parameter
        w = (obj["xmax"] - obj["xmin"]) / im_w              # The bounding box width is normalized to be measured relative to the image width
        h = (obj["ymax"] - obj["ymin"]) / im_h              # The bounding box height is normalized to be measured relative to the image height
        row = [w, h]
        BBoxDimensions.append(row)

BBoxDimensions = np.array(BBoxDimensions)                   # Convert python list to numpy array
print("Widths and heights transfered to array. Shape of array = ", BBoxDimensions.shape)

# Clustering the height and width data using different 'k' numbers of clusters.

kmax = 10                                                                                                               # The following loop will run the K-Means algorithm "kmax" - 1 times. 
fig = plt.figure(figsize=(7, 8))                                                                                        # A figure of 7 in x 8 in is created
meanDistance = np.zeros((kmax,1))                                                                                       # Array that will store the median IoU of all points in a cluster to their respective centers
results = {}

for k in range(2, kmax + 1):                                                                                            # The data will be clustered into 2, 3, ..., "kmax" clusters
    
    cluster_centers, cluster_labels, distances = kmeans.run_kmeans(BBoxDimensions, k)                                   # Running the K-Means algorithm on the "BBoxDImensions" data points, grouping them into 'k' clusters
    result = {"centers":    cluster_centers,
              "labels":     cluster_labels,
              "distances":  distances}
    results[k] = result

    meanDistance[k-2] = np.mean(distances[np.arange(distances.shape[0]),cluster_labels])                                # The distance of every point to its cluster center is retrieved. Then we compute the mean of all the returned values
    ax = fig.add_subplot(kmax/2, 2, k-1)                                                                                # Add a (kmax/2, 2) matrix of subplots, where the next plot will land on the "k-1" position.
    
    kmeans.plot_cluster_result(plt, cluster_centers, cluster_labels, 1 - float(meanDistance[k-2]), BBoxDimensions)      # The function plots and colors the different clusters created after running the algorithm.

plt.tight_layout()                                                                                                      # Execute this function to prevent overlap between the titles and axis titles of the subplots
plt.show()                                                                                                              # Show the final plot

# Determining the elbow point. After running the previous algorithm, the user must 
# establish the point in the following graph where an "elbow" forms. This is an adequate
# amount of clusters to retrieve 

plt.figure()
plt.plot(np.arange(2,kmax), [1 - meanDistance[k] for k in range(0,kmax-2)], "o-")                                       # x axis = numbers from 2 to kmax. y axis = 1 - meanDistance = meanIoU. 
plt.title("Elbow Curve\nTake Note of The 'N' where an Elbow Appears to Form")
plt.ylabel("Mean IoU")
plt.xlabel("Number of Clusters")
plt.show()


Elbow = input("Enter the amount of cluster N: ")                                                                        # Enter the "elbow point" where the slope of the graph appears to start changing most drastically
ANCHORS = results[int(Elbow)]["centers"]                                                                                # Extract the cluster centers that correspond to the elbow point selected and save them to ANCHORS
print("Predefined anchor box sizes: ", ANCHORS)                                                                         # For presentations sake, the ANCHORS are presented as a 2 column array
ANCHORS = np.squeeze(ANCHORS.reshape((1,-1)))                                                                           # "ANCHORS" needs to be a row vector. To do this, the array is reshapen to have a single row and then is "squeezed" to remove redundant dimensions


# INPUT ENCODING ***************************************************
print("\nTask: Input Encoding")
# Image is resized, its color channels are reordered and its data is 
# normalized (Normalization function must be defined previously)

def normalize(image):                                                           # Normalization function
    return image / 255.

TestImage = train_images[0]
InputEncoder = ImageReader(IMAGE_H = 416, IMAGE_W = 416, norm = image)          # ImageReader object. It resizes to 416x416 size and then normalizes the image
image, all_objs = InputEncoder.fit(TestImage)                                   # Re-scaling of boundary boxes and image

plt.imshow(image)
plt.title("image.shape={}".format(image.shape))
plt.show()

# OUTPUT ENCODING ***************************************************
print("\nTask: Output Encoding")
# Using "BestAnchorBoxFinder" to find the predefined anchor box shape that
# best fits a group of hypothetical boundary boxes detected in images. 

for i in range(0, len(ANCHORS), 2):
    print("Anchor box no.", i, ", w =", ANCHORS[i], "h =", ANCHORS[i+1])

BestAnchorFinder = BestAnchorBoxFinder(ANCHORS)

for w in range(1, 7, 2):
    w /= 10
    for h in range(1, 7, 2):
        h /= 10
        best_anchor, max_iou = BestAnchorFinder.find(w, h)
        print("BBox (w =", w, "h=", h, ") --> Best anchor index =", best_anchor, "iou =", max_iou) 