import cv2
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # img_test = cv2.imread('data/rtr-174-jarad-golgari-lich-lord.jpg')
    # img_test = cv2.imread('data/cn2-78-queen-marchesa.png')
    img_test = cv2.imread('data/c16-143-burgeoning.png')
    # img_test = cv2.imread('data/li38_handOfCards.jpg')
    # img_test = cv2.imread('data/pro_tour_side.png')
    img_gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow('original', img_test)
    cv2.imshow('threshold', img_thresh)

    kernel = np.ones((7, 7), np.uint8)
    img_dilate = cv2.dilate(img_thresh, kernel, iterations=1)
    # img_erode = cv2.erode(img_thresh, kernel, iterations=1)
    # img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('dilated', img_dilate)
    # cv2.imshow('eroded', img_erode)
    # cv2.imshow('opened', img_open)
    # cv2.imshow('closed', img_close)
    img_edge = cv2.Canny(img_dilate, 100, 200)
    cv2.imshow('edge', img_edge)

    lines = cv2.HoughLines(img_edge, 1, np.pi / 180, 200)
    if lines is not None:
        img_hough = cv2.cvtColor(img_dilate.copy(), cv2.COLOR_GRAY2BGR)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('hough', img_hough)
    else:
        print('Hough couldn\'t find any lines')
    '''
    lines = cv2.HoughLinesP(img_edge, 1, np.pi / 180, 200, 50, 100)
    if lines is not None:
        img_hough = cv2.cvtColor(img_dilate.copy(), cv2.COLOR_GRAY2BGR)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_hough, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.imshow('hough', img_hough)
    else:
        print('Hough couldn\'t find any lines')

    img_contour, contours, hierchy = cv2.findContours(img_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_contour = cv2.cvtColor(img_contour, cv2.COLOR_GRAY2BGR)
    if len(contours) > 0:
        cv2.drawContours(img_contour, contours, -1, (0, 0, 255), 1)

    cv2.imshow('contours', img_contour)
    '''
    cv2.waitKey(0