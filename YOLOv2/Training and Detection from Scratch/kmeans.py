import numpy as np
import seaborn as sns
import itertools

def IoU(single_box, box_array):
    '''
    Determine the IoU between a single bounding box (single_box) and "box_array" (N, 2)
    containing the dimensions of multiple bounding boxes.

    Parameters
    -----------

    single_box (np.array: 1,2)
        Array containing the w and h of a test bounding box

    box_array (np.array: N,2)
        Array containing th width and height of multiple bounding boxes. Each 
        row of the array corresponds to one bounding box.

    Outputs
    ---------

    iou (np.array: N,1)
        Intersection over union between the single_box and each row of the 
        "box_array" (In other words, the IoU between the single box and each 
        data point in the box_array).

    '''

    x = np.minimum(box_array[:, 0], single_box[0])          # (N, 1) array containing the min(width) after comparing the single_box width and the width of each of the (N) data points
    y = np.minimum(box_array[:, 1], single_box[1])          # (N, 1) array containing the min(height) after comparing the single_box height and the height of each of the (N) data points

    intersection = x * y                                    # Element wise multiplication of every min(width) and min(height)
    box_area = single_box[0] * single_box[1]                # Single box area
    cluster_area = box_array[:, 0] * box_array[:, 1]        # Area of all the bounding boxes contained in the array
    total_area = box_area + cluster_area                    # Sum of both areas
    union = total_area - intersection                       # Union = area both boxes - the intersection between both boxes

    iou = intersection / union

    return iou

def run_kmeans(boxes, k, mean = np.median, seed = 1):
    '''
    Method that utilizes the K-means clustering algorithm to group the bounding box dimension data 
    into recognizable groups. Given a group of points in space (In this case, two dimensional points)
    the algorithm: 
    
        1. Assigns "k" random points from the data as cluster centers
        2. Calculates the distance of every other point to each cluster center
        3. Determines the smallest "point to cluster center" distance per point
        4. Assigns a "cluster label" to each point based on the previous result
        5. Cluster centers are updated: Median of all points with the same label
        6. Repeat til no "cluster label" updates occur

    Parameters
    -----------

    boxes (np.array: N,2)
        Array with N rows (N samples) and 2 columns (Width and height of 
        bounding box)

    k (int)
        Number of clusters to generate

    dist (func)
        Method to extract the "mean" of all points. Default value is "np.median"

    seed (int)
        Seed to generate the initial random selection of cluster centers

    Outputs
    --------

    cluster_centers (np.array: k,2)
        The center point of each cluster center as a width and height pair

    cluster_labels (np.array: N,1)
        Array of "cluster labels", one for each point given in boxes

    distances (np.array: N, k)
        Distance of each point to every cluster center.

    '''

    rows = boxes.shape[0]                                                       # Extracts the number of rows in the "boxes" vector

    distances = np.empty((rows, k))                                             # Distance of every point (Each row) to every cluster center (Each column)
    last_cluster_labels = np.zeros((rows,1))                                    # Vector containing all the "cluster labels" for every point
    cluster_labels = np.zeros((rows,1))                                         # Vector containing all the "cluster labels" for every point

    np.random.seed(seed)                                                        # Make random generations predictable by using a constant seed.

    cluster_centers = boxes[np.random.choice(rows, k, replace=False)]           # Assign "k" random points from "boxes" as cluster centers

    while True:

        # Step 1: Allocate each item to the closest cluster centers. In standard K-Means the
        # euclidean distance is used. However, in this case the distance is calculated using
        # 1 - IoU (Between cluster center and all the other bounding boxes)
        for i in range(k):
            distances[:, i] = 1 - IoU(cluster_centers[i], boxes)                # Column "i" of "distance" = 1 - IoU between all boundary boxes and cluster center "i"
        
        cluster_labels = np.argmin(distances, axis = 1)                         # For each row returns the number of column that generated the smaller distance
        

        if (last_cluster_labels == cluster_labels).all():                       # If there was no change in cluster asignments, end the algorithm
            break
        
        # Step 2: Update the cluster centers by calculating the mean of all the 
        # points that correspond to the same cluster.
        for i in range(k):
            cluster_centers[i] = mean(boxes[cluster_labels == i], axis=0)       # Get the median of all the bounding boxes (Widths and heights) that correspond to the same cluster "i"
        
        last_cluster_labels = cluster_labels                                    # All the cluster labels are saved as "last_clusters"         
    
    return cluster_centers, cluster_labels, distances

def plot_cluster_result(plt, cluster_centers, cluster_labels, meanIoU, box_data):

    palette = itertools.cycle(sns.xkcd_rgb.values())                    # Import the seaborn palette xkcd_rgb. cycle() creates an iterator that returns a different color every iteration

    for i in np.unique(cluster_labels):                                 # np.unique returns an array of non-repeating "cluster labels". The for loop goes through label 1, 2, 3...  

        pick = (cluster_labels==i)                                      # "pick" consists of all the points that  
        plt.plot(box_data[pick,0], box_data[pick,1],"p",                # All the rows asociated with cluster "i" will be plotted as a point 
                 color=next(palette),
                 alpha=0.5)
        plt.text(cluster_centers[i,0],                                  # A "C1, C2, C3,..." label appears on each cluster center
                 cluster_centers[i,1],
                 "c{}".format(i),
                 fontsize=10, color="red")
        plt.title("No. Clusters = {}     Mean IoU = {:5.4}".format(cluster_centers.shape[0], meanIoU), fontsize=8)
        plt.xlabel("Width", fontsize=8)
        plt.ylabel("Height", fontsize=8)