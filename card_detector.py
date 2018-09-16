import cv2
import numpy as np
import pandas as pd
import math
from screeninfo import get_monitors


def detect_a_card(img, thresh_val=80, blur_radius=None, dilate_radius=None, min_hyst=80, max_hyst=200,
                  min_line_length=None, max_line_gap=None, debug=False):
    dim_img = (len(img[0]), len(img)) # (width, height)
    # Intermediate variables

    # Default values
    if blur_radius is None:
        blur_radius = math.floor(min(dim_img) / 100 + 0.5) // 2 * 2 + 1  # Rounded to the nearest odd
    if dilate_radius is None:
        dilate_radius = math.floor(min(dim_img) / 67 + 0.5)
    if min_line_length is None:
        min_line_length = min(dim_img) / 10
    if max_line_gap is None:
        max_line_gap = min(dim_img) / 10

    thresh_radius = math.floor(min(dim_img) / 20 + 0.5) // 2 * 2 + 1  # Rounded to the nearest odd

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Median blur better removes background textures than Gaussian blur
    img_blur = cv2.medianBlur(img_gray, blur_radius)
    # Truncate the bright area while detecting the border
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, thresh_radius, 20)
    #_, img_thresh = cv2.threshold(img_blur, thresh_val, 255, cv2.THRESH_TRUNC)

    # Dilate the image to emphasize thick borders around the card
    kernel_dilate = np.ones((dilate_radius, dilate_radius), np.uint8)
    #img_dilate = cv2.dilate(img_thresh, kernel_dilate, iterations=1)
    img_dilate = cv2.erode(img_thresh, kernel_dilate, iterations=1)

    img_contour = img_dilate.copy()
    _, contours, _ = cv2.findContours(img_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contour = cv2.cvtColor(img_contour, cv2.COLOR_GRAY2BGR)
    img_contour = cv2.drawContours(img_contour, contours, -1, (128, 128, 128), 1)
    card_found = contours is not None
    print(len(contours))
    print([len(contour) for contour in contours])

    # find the biggest area
    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)
    # draw the book contour (in green)
    img_contour = cv2.drawContours(img_contour, [c], -1, (0, 255, 0), 1)

    # Canny edge - low minimum hysteresis to detect glowed area,
    # and high maximum hysteresis to compensate for high false positives.
    img_canny = cv2.Canny(img_dilate, min_hyst, max_hyst)
    #img_canny = img_dilate
    # Apply Hough transformation to detect the edges
    detected_lines = cv2.HoughLinesP(img_dilate, 1, np.pi / 180, threshold=60,
                                     minLineLength=min_line_length,
                                     maxLineGap=max_line_gap)
    card_found = detected_lines is not None
    print(len(detected_lines))

    if card_found:
        if debug:
            img_hough = cv2.cvtColor(img_dilate.copy(), cv2.COLOR_GRAY2BGR)
            for line in detected_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_hough, (x1, y1), (x2, y2), (0, 0, 255), 1)
    elif not debug:
        print('Hough couldn\'t find any lines')

    # Debug: display intermediate results from various steps
    if debug:
        img_blank = np.zeros((len(img), len(img[0]), 3), np.uint8)
        img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
        img_dilate = cv2.cvtColor(img_dilate, cv2.COLOR_GRAY2BGR)
        #img_canny = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
        if not card_found:
            img_hough = img_blank

        # Append all images together
        img_row_1 = np.concatenate((img, img_thresh), axis=1)
        img_row_2 = np.concatenate((img_contour, img_hough), axis=1)
        img_result = np.concatenate((img_row_1, img_row_2), axis=0)

        # Resize the final image to fit into the main monitor's resolution
        screen_size = get_monitors()[0]
        resize_ratio = max(len(img_result[0]) / screen_size.width, len(img_result) / screen_size.height, 1)
        img_result = cv2.resize(img_result, (int(len(img_result[0]) // resize_ratio),
                                             int(len(img_result) // resize_ratio)))
        cv2.imshow('Result', img_result)
        cv2.waitKey(0)

    # TODO: output meaningful data
    return card_found

def main():
    img_test = cv2.imread('data/li38_handOfCards.jpg')
    card_found = detect_a_card(img_test,
                               #dilate_radius=5,
                               #thresh_val=100,
                               #min_hyst=40,
                               #max_hyst=160,
                               #min_line_length=50,
                               #max_line_gap=100,
                               debug=True)
    if card_found:
        return
    return
    for dilate_radius in range(1, 6):
        for min_hyst in range(50, 91, 10):
            for max_hyst in range(180, 119, -20):
                print('dilate_radius=%d, min_hyst=%d, max_hyst=%d: ' % (dilate_radius, min_hyst, max_hyst),
                      end='', flush=True)
                card_found = detect_a_card(img_test, dilate_radius=dilate_radius,
                                           min_hyst=min_hyst, max_hyst=max_hyst, debug=True)
                if card_found:
                    print('Card found')
                else:
                    print('Not found')

if __name__ == '__main__':
    main()
