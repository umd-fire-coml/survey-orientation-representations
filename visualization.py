import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math
from matplotlib.path import Path
import matplotlib.patches as patches
from typing import List

## Draw 3D orientation visualization

def IOU(box1, box2):
    y_overlap = max(0, min(box1[1]+box1[3], box2[1]+box2[3]) - max(box1[1], box2[1])); #Leftmost right edge minus rightmost left edge 
    x_overlap = max(0, min(box1[0]+box1[2], box2[0]+box2[2]) - max(box1[0], box2[0])); #Highest bottom minus lowest top
    interArea = x_overlap * y_overlap;
    unionArea = (box1[3]) * (box1[2]) + (box2[3]) * (box2[2]) - interArea
    iou = interArea / unionArea
    return iou

# Overlay an arrow of a given instance (from alpha_pred) on the image img
def get_3d_arrow_on_img_from_batch_alpha_pred(calibration_file_path:str, batch_kitti_label_line: List[str], img, batch_alpha_pred: List[float], batch_dim_pred: List[List[float]] =None):
    for i, line in enumerate(batch_kitti_label_line):
        if batch_dim_pred is None:
            img = __draw_arrow_in_3d_from_single_alpha_pred(calibration_file_path, line, img, single_alpha_pred=batch_alpha_pred[i])
        else:
            img = __draw_arrow_in_3d_from_single_alpha_pred(calibration_file_path, line, img, single_alpha_pred=batch_alpha_pred[i], dim_pred=batch_dim_pred[i])
    return img

def __draw_arrow_in_3d_from_single_alpha_pred(calibration_file_path, single_kitti_label_line, img, single_alpha_pred, dim_pred=None):
             
    line = single_kitti_label_line
    line = line.strip().split(' ')
    x = float(line[11])
    y = float(line[12])
    z = float(line[13])
    single_rot_y_pred = single_alpha_pred + math.atan2(x, z) # the formula from gs 3d
    return __draw_arrow_in_3d_from_single_rot_y_pred(calibration_file_path, single_kitti_label_line, img, single_rot_y_pred, dim_pred)

# Overlay an arrow of a given instance (from rot_y_pred) on the image img
def __draw_arrow_in_3d_from_single_rot_y_pred(calibration_file_path, single_kitti_label_line, img, single_rot_y_pred, dim_pred=None):
    """
    Creates display for ground truth angle and predication angle
    
    calibration - str directory to calibration file
    kitti_label - str in the format of kitti label for the desired instance
    img         - read in image using plt
    """
    for line in open(calibration_file_path):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3,4))
            
    line = single_kitti_label_line
    line = line.strip().split(' ')

    dims   = np.asarray([float(number) for number in line[8:11]])
    center = np.asarray([float(number) for number in line[11:14]])
    gt_rot_y = float(line[14])

    box_3d = []
    pred_3d = []
    count = 0

    for i in [1,-1]:
        for j in [1,-1]:
            for k in [0,1]:
                point = np.copy(center)

                point[0] = center[0] + i * dims[1]/2 * np.cos(-gt_rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.cos(-gt_rot_y)
                point[2] = center[2] + i * dims[1]/2 * np.sin(-gt_rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.sin(-gt_rot_y)                  
                point[1] = center[1] - k * dims[0]

                count += 1

                point = np.append(point, 1)
                point = np.dot(cam_to_img, point)

                point = point[:2]/point[2]
                point = point.astype(np.int16)

                box_3d.append(point)
    for i in range(4):
        point_1_ = box_3d[2*i]
        point_2_ = box_3d[2*i+1]
        disp = cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0,255,0), 2)
    
    for i in range(8):
        point_1_ = box_3d[i]
        point_2_ = box_3d[(i+2)%8]
        disp = cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0,255,0), 2)
    if dim_pred is not None:
        count = 0
        for i in [1,-1]:
            for j in [1,-1]:
                for k in [0,1]:
                    point = np.copy(center)

                    point[0] = center[0] + i * dim_pred[1]/2 * np.cos(-single_rot_y_pred+np.pi/2) + (j*i) * dim_pred[2]/2 * np.cos(-single_rot_y_pred)
                    point[2] = center[2] + i * dim_pred[1]/2 * np.sin(-single_rot_y_pred+np.pi/2) + (j*i) * dim_pred[2]/2 * np.sin(-single_rot_y_pred)                  
                    point[1] = center[1] - k * dim_pred[0]

                    count += 1

                    point = np.append(point, 1)
                    point = np.dot(cam_to_img, point)

                    point = point[:2]/point[2]
                    point = point.astype(np.int16)

                    pred_3d.append(point)
        for i in range(4):
            point_1_ = pred_3d[2*i]
            point_2_ = pred_3d[2*i+1]
            disp = cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255,0,0), 5)

        for i in range(8):
            point_1_ = pred_3d[i]
            point_2_ = pred_3d[(i+2)%8]
            disp = cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255,0,0), 5)

    center = np.append(center, 1)
    endpoint = [0, 0, 0, 0]
    gt_endpoint = [0, 0, 0, 0]

    endpoint = make_endpoint(endpoint, center, dims, single_rot_y_pred)
    gt_endpoint = make_endpoint(gt_endpoint, center, dims, gt_rot_y)
    
    center = convert_to_2d(center, cam_to_img)
    endpoint = convert_to_2d(endpoint, cam_to_img)
    gt_endpoint = convert_to_2d(gt_endpoint, cam_to_img)

    disp = cv2.arrowedLine(img, center, endpoint, color = (255, 0, 0), thickness = 2, tipLength = .5)
    disp = cv2.arrowedLine(img, center, gt_endpoint, color = (0, 255, 0), thickness = 2, tipLength = .5)
    
    return disp


## Draw Birds Eye view visualization



def convert_to_2d(point, cam_to_img):
    """
    Used in make_arrow
    """
    #point is 4 element array
    point = np.dot(cam_to_img, point)
    point = point[:2]/point[2]
    point = point.astype(np.int16)
    point = tuple(point)
    
    return point


def make_endpoint(point, center, dims, rot_y):
    """
    Used in make_arrow
    """
    point[0] = center[0] + 0 * np.cos(-rot_y+np.pi/2) + (dims[2]/2) * np.cos(-rot_y)
    point[2] = center[2] + 0 * np.sin(-rot_y+np.pi/2) + (dims[2]/2) * np.sin(-rot_y)                  
    point[1] = center[1]
    
    return point

# TODO: add directional line indicating the front face. e.g. https://arxiv.org/pdf/1902.06326.pdf
def compute_birdviewbox(line, shape, scale):
    npline = [np.float64(line[i]) for i in range(1, len(line))]
    h = npline[7] * scale
    w = npline[8] * scale
    l = npline[9] * scale
    x = npline[10] * scale
    y = npline[11] * scale
    z = npline[12] * scale
    rot_y = npline[13]

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                  [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2


    x_corners += -w / 2
    z_corners += -l / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(shape/2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    return np.vstack((corners_2D, corners_2D[0,:]))

def compute_center_arrow(line, shape,scale):
    npline = [np.float64(line[i]) for i in range(1, len(line))]
    h = npline[7] * scale
    w = npline[8] * scale
    l = npline[9] * scale
    x = npline[10] * scale
    y = npline[11] * scale
    z = npline[12] * scale
    rot_y = npline[13]
    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                [np.sin(rot_y), np.cos(rot_y)]])
    x_corners = [l/2, 1.5*l]  # -l/2
    z_corners = [w/2, w/2]  # -w/2
    x_corners += -w / 2
    z_corners += -l / 2
    t = np.array([x, z]).reshape(1, 2).T
    corners_2D = np.array([x_corners, z_corners])
    
    corners_2D = R.dot(corners_2D)
    corners_2D = t - corners_2D
    corners_2D[0] += int(shape/2)

    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T
    return np.vstack((corners_2D, corners_2D[0,:])) # I have no idea why we need a line back but path works in mysterious patch

def draw_birdeyes(ax2, line_gt, line_p, shape):  
    # shape = 900
    scale = 15

    pred_corners_2d = compute_birdviewbox(line_p, shape, scale)
    pred_arrow_2d = compute_center_arrow(line_p, shape, scale)
    gt_corners_2d = compute_birdviewbox(line_gt, shape, scale)
    gt_arrow_2d = compute_center_arrow(line_gt, shape, scale)

    codes = [Path.LINETO] * gt_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(gt_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='orange', label='gt_box')
    ax2.add_patch(p)
    
    codes = [Path.LINETO] * gt_arrow_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(gt_arrow_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='red', label='gt_arrow')
    ax2.add_patch(p)
    
    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='green', label='pred_box')
    ax2.add_patch(p)
    
    codes = [Path.LINETO] * pred_arrow_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_arrow_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='m', label='pred_arrow')
    ax2.add_patch(p)
    



def draw_box3D(image, kitti_label, calib_file, color = (0, 255, 0)):
    #image is numpy image
    #calib_file is calib file associated with image
    #kitti_label is image labels in the style of Kitti
    #color is desired color
    
    for line in open(calib_file):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3,4))
            
    for line in open(kitti_label):
        line = line.strip().split(' ')

        dims   = np.asarray([float(number) for number in line[8:11]])
        center = np.asarray([float(number) for number in line[11:14]])
        rot_y  = float(line[14])

        box_3d = []
        
        count = 0

        for i in [1,-1]:
            for j in [1,-1]:
                for k in [0,1]:
                    point = np.copy(center)

                    point[0] = center[0] + i * dims[1]/2 * np.cos(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.cos(-rot_y)
                    point[2] = center[2] + i * dims[1]/2 * np.sin(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.sin(-rot_y)                  
                    point[1] = center[1] - k * dims[0]
                

                    print(count, i, j, k, end = " ")
                    print("     POINT:",point,end = " ")
                    print("      DIMS:", dims, end=" ")
                    count += 1
                    
                    point = np.append(point, 1)
                    point = np.dot(cam_to_img, point)

                    point = point[:2]/point[2]
                    point = point.astype(np.int16)
                    
                    print(point)
                    box_3d.append(point)

        for i in range(4):
            point_1_ = box_3d[2*i]
            point_2_ = box_3d[2*i+1]
            cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), color, 5)
            #print("line")
            #print(point_1_, point_2_)

        for i in range(8):
            point_1_ = box_3d[i]
            point_2_ = box_3d[(i+2)%8]
            cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), color, 5)
            
    return image

class Bev_viz(object):
    PIXEL_RATIO = 20
    #labels is supposed to be a list of str
    
    def __init__(self, ground_truths=None, predictions=None):
        '''ground_truths is a list of space separated lines in Kitti label format
        predictions is a list of space separated lines in Kitti label format'''
    #Accepts ground truths and predictions as a list or a string file name  
    
        if ground_truths is None:
            self.ground_truths = []
        elif type(ground_truths) is list:
            self.ground_truths = ground_truths
        elif type(ground_truths) is str:
            with open(ground_truths, 'r') as file:
                self.ground_truths = file.readlines()
                self.ground_truths = list(map(str.strip, self.ground_truths)) # removes /r/n at end of line
                
        if predictions is None:
            self.predictions = []
        elif type(predictions) is list:
            self.predictions = predictions
        elif type(predictions) is str:
            with open(predictions, 'r') as file:
                self.predictions = file.readlines()
                self.predictions = list(map(str.strip, self.ground_truths))
    
    def add_gt(self, label):
        #Add gt label
        if type(label) is list:
            self.ground_truths += label
        elif type(label) is str:
            with open(label, 'r') as file:
                temp = file.readlines()
                temp = list(map(str.strip, temp)) # removes /r/n at end of line
                self.ground_truths += temp
            
    def add_pred(self, label):
        #Add pred label
        if type(label) is list:
            self.predictions += label
        elif type(label) is str:
            with open(label, 'r') as file:
                temp = file.readlines()
                temp = list(map(str.strip, temp)) # removes /r/n at end of line
                self.predictions += temp
    
    def label_to_bbox(label):
        #get info
        info = label.split()
        print(info)
        x = float(info[11])
        z = float(info[13])
        w = float(info[9])
        l = float(info[10])
        rot_y = -1.0*float(info[14])
        #Axis adjustment
        rot_y -= math.pi/2
        
        #Define rotation matrix and box before rotation
        rot_matrix = np.array([[np.cos(rot_y), -np.sin(rot_y)], [np.sin(rot_y), np.cos(rot_y)]])
        lw_box = np.array([[-l/2, l/2, l/2, -l/2], [-w/2, -w/2, w/2, w/2]])
        scaled_lw_box = np.multiply(lw_box, Bev_viz.PIXEL_RATIO) # scale box meters to pixels
        
        #Calculate points for rotated box in the top down view (xz plane)
        rotated_box = np.matmul(rot_matrix, scaled_lw_box)
        # element wise shift by xz location
        rotated_box = np.add(rotated_box, np.array([[z * Bev_viz.PIXEL_RATIO],[x * Bev_viz.PIXEL_RATIO]]))
        rotated_box = np.round(rotated_box)
        rotated_box = rotated_box.astype(int)
        
        # shape of rotated_box is (2, 4)
        return rotated_box

    # Update to work if either ground_truths=None or predictions=None 
    def draw_BEV(self, gt_color=(0, 255, 0), pred_color=(255, 0, 0)):
        LINE_RATIO = 225 # scale line thickness by the world length dimension
        Y_MARGIN = 250
        #Z is the X-axis of the image, while X is the Y-axis of the image
        
        #Get boxes from labels
        gt_bboxes = np.array(list(map(Bev_viz.label_to_bbox, self.ground_truths)))
        pred_bboxes = np.array(list(map(Bev_viz.label_to_bbox, self.predictions)))
        
        viewpoint = [0,0]
        
        #Deal with negatives, get the z min of all boxes
        check_z_min = min(np.amin(gt_bboxes[:,0,:]), np.amin(pred_bboxes[:,0,:]))
        
        # shift all z positions forward such that all of them fit within z-bottom viewpoint
        if check_z_min < 0:
            bias = abs(check_z_min)
            gt_bboxes[:,0,:] += bias
            pred_bboxes[:,0,:] += bias
            viewpoint[0] = bias
        
        #Determine size of image by taking the maximum Z and X and using that a reference for length
        image_width = max(np.amax(gt_bboxes[:,0,:]), np.amax(pred_bboxes[:,0,:])) + Y_MARGIN # get highest z value for image width
        check_x_max = max(np.amax(np.absolute(gt_bboxes[:,1,:])), np.amax(np.absolute(pred_bboxes[:,1,:]))) # get highest x value
        image_height = (2 * check_x_max) + 30
        
        # Shift the x locations to y pixel positions such that all pixel positions are positive
        gt_bboxes[:,1,:] += int(image_height / 2)
        pred_bboxes[:,1,:] += int(image_height / 2)
        viewpoint[1] += int(image_height / 2)
        
        #Create image
        image = np.zeros((image_height, image_width, 3))
        
        #Draw boxes on image
        for i in range(len(gt_bboxes)):
            for k in range(3):
                cv2.line(image, (gt_bboxes[i][0,k], gt_bboxes[i][1,k]), (gt_bboxes[i][0,k+1], gt_bboxes[i][1,k+1]), gt_color, math.ceil(image_width / LINE_RATIO))
            #Draw last line
            cv2.line(image, (gt_bboxes[i][0,3], gt_bboxes[i][1,3]), (gt_bboxes[i][0,0], gt_bboxes[i][1,0]), gt_color, math.ceil(image_width / LINE_RATIO))
            
            #Draw class label
            cv2.putText(image, self.ground_truths[i].split()[0][0:3], (gt_bboxes[i][0,1], gt_bboxes[i][1,1]), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,255), 15, lineType=cv2.LINE_AA)
            
        for i in range(len(pred_bboxes)):
            for k in range(3):
                cv2.line(image, (pred_bboxes[i][0,k], pred_bboxes[i][1,k]), (pred_bboxes[i][0,k+1], pred_bboxes[i][1,k+1]), pred_color, math.ceil(image_width / LINE_RATIO))
            #Draw last line
            cv2.line(image, (pred_bboxes[i][0,3], pred_bboxes[i][1,3]), (pred_bboxes[i][0,0], pred_bboxes[i][1,0]), pred_color, math.ceil(image_width / LINE_RATIO))
        
        #Draw viewpoint
        cv2.arrowedLine(image, (viewpoint[0], viewpoint[1]), ((math.ceil(viewpoint[0] + image_width/20), viewpoint[1])), (255,255,255), math.ceil(image_width / LINE_RATIO) + 2, tipLength=0.4)
            
        return image
    
    def display(self):
        plt.imshow(self.draw_BEV())