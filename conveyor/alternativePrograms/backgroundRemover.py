import cv2
import numpy as np
import imutils

def removeBackground( image ):
    
    # display the original
    #cv2.imshow('1-original-image', image)

    # create a grayscale version of the image
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('2-grayscale-image', image_gray)

    # apply a blur (in addition to canny blur)
    image_blurred = cv2.blur(image_gray, (5, 5))
    #cv2.imshow('3-blurred-image', image_blurred)

    # grab edges- convert to contours
    edges = cv2.Canny(image_blurred, 30, 120)
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    
    # create a copy of the original image
    contoured_image = image.copy()

    # draw contours
    cv2.drawContours(contoured_image, contours, -1, (255, 255, 255), 4)
    #cv2.imshow('4-contoured-image', contoured_image)

    # apply a threshold
    ret,threshold_image = cv2.threshold(contoured_image, 250, 255, cv2.THRESH_BINARY)
    #cv2.imshow('5-threshold-image', threshold_image)
    
    # copy original image
    threshold_copy = threshold_image.copy()

    # find image dimensions and create a mask
    image_height = image.shape[0]
    image_width = image.shape[1]
    image_mask = np.zeros((image_height+2, image_width+2), np.uint8)

    # flood area between borders of image and outside of contours with white
    cv2.floodFill(threshold_copy, image_mask, (0,0), (255,255,255))
    #cv2.imshow('6-flooded-image', threshold_copy)

    # invert colors
    inverted_threshold_image = cv2.bitwise_not(threshold_copy)
    #cv2.imshow('7-inverted-image', inverted_threshold_image)

    # OR with threshold image to create a black background
    inverted_threshold_image = inverted_threshold_image | threshold_image
    #cv2.imshow('8-inverted-image', inverted_threshold_image)

    # invert all to get final white background
    final_background = cv2.bitwise_not(inverted_threshold_image)

    # OR final image to get final image on black background
    final_image = final_background | image
    #cv2.imshow('9-final-image', final_image)

    # return final image
    return final_image
