import cv2
import numpy as np
import pandas as pd

if __name__ == '__main__':
    #img_test = cv2.imread('data/rtr-174-jarad-golgari-lich-lord.jpg')
    #img_test = cv2.imread('data/cn2-78-queen-marchesa.png')
    #img_test = cv2.imread('data/c16-143-burgeoning.png')
    #img_test = cv2.imread('data/handOfCards.jpg')
    img_test = cv2.imread('data/pro_tour_side.png')
    img_gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
    #cv2.imshow('original', img_test)
    cv2.imshow('threshold', img_thresh)

    kernel = np.ones((4, 4), np.uint8)
    img_dilate = cv2.dilate(img_thresh, kernel, iterations=1)
    #img_erode = cv2.erode(img_thresh, kernel, iterations=1)
    #img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('dilated', img_dilate)
    #cv2.imshow('eroded', img_erode)
    #cv2.imshow('opened', img_open)
    #cv2.imshow('closed', img_close)

    cv2.waitKey(0)
