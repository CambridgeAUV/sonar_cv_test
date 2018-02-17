#!/usr/bin/env python

import cv2
import numpy as np
import random
import math
import rospy
from std_msgs.msg import Float32MultiArray
import copy

def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

def get_dis_to_line(point, line_pt_1, line_pt_2):
    # Given a point and two points on a line (line_pt_1 and line_pt_2) returns
    # the shortest distance from the point to the line through those two points.
    # print ("Inputs to get_dis_to_line: ", point, " ", line_pt_1, " ", line_pt_2)


    line_pt_1.append(0)
    line_pt_2.append(0)
    point.append(0)
    line_vector = np.array(line_pt_2) - np.array(line_pt_1)
    perp_line = np.cross(line_vector, [0, 0, 1])
    unit_perp_line = perp_line / np.linalg.norm(perp_line)

    dis_to_line = np.dot(unit_perp_line, np.array(line_pt_2) - np.array(point))

    perp_line_with_dis = unit_perp_line * dis_to_line
    # cv2.line(img, (point[0], point[1]), (point[0] + int(perp_line_with_dis[0]), point[1] + int(perp_line_with_dis[1])), (255, 0, 0), 2)
    return abs(dis_to_line)

def get_two_points_on_a_line(line):
    #Given a line of the form [rho, theta] returns a list of the form
    #   [x1, y1, x2, y2] where x1,y1 and x2,y2 are two points on the line
    theta = line[1]
    rho = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    length = 2000
    x1 = int(x0 + length*(-b))
    y1 = int(y0 + length*(a))
    x2 = int(x0 - length*(-b))
    y2 = int(y0 - length*(a))
    return [x1, y1, x2, y2]

def order_lines (inp_set, image_height):
    # Orders inp_set by each line's distance to the point (0, image_height/2)
    # Expecting lines of the form [rho, theta]

    lines = copy.deepcopy(inp_set)
    original_lines_length = len(lines)

    ordered_new_set = []
    index_of_closest_line = 0
    closest_distance = 99999

    for c in range(original_lines_length): 

        for i in range(len(lines)):
            two_points_on_line = get_two_points_on_a_line(lines[i])
            dis = get_dis_to_line([0,image_height/2], two_points_on_line[0:2], two_points_on_line[2:4])
            if (dis < closest_distance):
                closest_distance = dis
                index_of_closest_line = i

        ordered_new_set.append(lines[index_of_closest_line])
        del lines[index_of_closest_line]

        closest_distance = 99999
        index_of_closest_line = 0

    return ordered_new_set



def get_grouped_lines(lines):
    if not(lines is None):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, grouped_lines = cv2.kmeans(lines, 2,None,  criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return grouped_lines
    else:
        print("There were no lines found")
        return []

def get_river_sides(lines):
    # Apply RANSAC to find the river edges

    # Choose two lines at random
    # If their r's  are too similar, disregard the sample (since two lines of the same side have been chosen)
    # Error is the sum of the squares of the angle differences to other lines on this side of the river
    # Lines are judged to be on the same side of the rive if their r's are similar

    #Lines format is rho 0 theta 1 

    r_threshold = 30
    iterations = 100

    if lines is None:
        return []

    random.seed()


    # Some of the lines have a negative rho and theta which is approx pi
    # First go through and change these so everyone is consistent
    for i in range(len(lines)):
        if lines[i][0][0] < 0:
            lines[i][0][0] = 0 - lines[i][0][0]
            lines[i][0][1] -= math.pi

    lowest_error = 100000
    best_line_pair = []
    for i in range(iterations):
        sample = []
        sample.append(lines[random.randint(0, len(lines) - 1)][0])
        sample.append(lines[random.randint(0, len(lines) - 1)][0])

        # Disregard sample if lines are on the same side of the river
        if abs(sample[0][0] - sample [1][0]) < r_threshold:
            continue

        # Calculate the squared error
        error_total = 0
        for line in lines:
            line = line[0]
            for sample_line in sample:
                if (abs(line[0] - sample_line[0]) < r_threshold):
                    error_total += pow(line[0] - sample_line[0], 2)

        if error_total < lowest_error:
            lowest_error = error_total
            best_line_pair = sample
        

    return best_line_pair
    

def filter(img):

    kernel = np.ones((3,3),np.float32)/9
    img = cv2.filter2D(img,-1,kernel)

    img = cv2.Canny(img,0,80)

    # _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # convert to skeleton
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    skel = np.zeros(img.shape).astype('uint8')
    
    while True:

        temp = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.bitwise_not(temp)
        temp = cv2.bitwise_and(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = cv2.erode(img, element)
        
        if np.sum(img) == 0:
            break
    img = skel

    return img

cap = cv2.VideoCapture('/home/andrew/code/sonar_cv_test/sonar_vid_crop.avi')
publisher = rospy.Publisher('river_sides', Float32MultiArray, queue_size = 10)
# The format of the message is: distance to left side, distance to right side,
# angle of left side (CW from vertical, rads), angle of right side (CW from vertical, rads).
rospy.init_node('sonar_cv', anonymous = True)
rate = rospy.Rate(10)

old_lines = []
while(cap.isOpened()):

    ret, raw_img = cap.read()
    if raw_img.any():
        cv2.imshow('raw', raw_img)

    # cut off top 30px, which is text
    cut_px = 30
    raw_img = raw_img[cut_px:, :]

    img = np.array(raw_img)
    # Threshold the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = filter(img)

    # Find Hough lines
    lines = cv2.HoughLines(img, 1, np.pi/180, threshold = 40)

    lines = get_river_sides(lines)

    # Draw the submarine on the image
    sub_pos = (int(len(raw_img[0])/2), len(raw_img) - 20)
    cv2.circle(raw_img, sub_pos, 5, (100, 255, 0), -1)

    # Draw the lines on the original image and find the distance to the lines from the submarine

    dis_to_lines = []
    angle_of_lines = []


    if not(old_lines == None):
        lines = order_lines(lines, len(raw_img))


    # text_pos_counter = 0
    if lines is not None:
        for line in lines:
            # line = line[0]
            theta = line[1]
            rho = line[0]

            angle_of_lines.append(theta)

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            length = 2000
            x1 = int(x0 + length*(-b))
            y1 = int(y0 + length*(a))
            x2 = int(x0 - length*(-b))
            y2 = int(y0 - length*(a))

            # x1, y1, x2, y2 = tuple(line[0])
            cv2.line(raw_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

            dis_to_line = get_dis_to_line([sub_pos[0], sub_pos[1]], [x1, y1], [x2, y2])
            print("dis_to_line: ", dis_to_line)
            dis_to_lines.append(dis_to_line)
            # # cv2.putText(raw_img, "Line " + str(text_pos_counter) + " dis: " + str(dis_to_line), (100 , 200 + 100 * text_pos_counter), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            # text_pos_counter += 1

    message_packet = Float32MultiArray(data=dis_to_lines + angle_of_lines)

    publisher.publish(message_packet)

    print "********"



    # cv2.imshow('video',cv2.resize(raw_img, ( int(len(raw_img[0]) * 0.5), int(len(raw_img) * 0.5)) ))
    cv2.imshow('filtered', img)
    cv2.imshow('video', raw_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Breaking loop")
        break

    rate.sleep()
cap.release()
cv2.destroyAllWindows()


print("End of program")
