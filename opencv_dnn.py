import ast
import collections
import cv2
import imagehash as ih
import numpy as np
from operator import itemgetter
import os
import pandas as pd
from PIL import Image
import time

from config import Config
import fetch_data


"""
As of the current version, the YOLO network has been removed from this code during optimization.
It was found out that YOLO was adding too much processing delay, and the benefits from using it couldn't justify
such heavy cost.
If you're interested to see the implementation using YOLO, please check out the previous commit:
https://github.com/hj3yoo/mtg_card_detector/tree/dea64611730c84a59c711c61f7f80948f82bcd31 
"""


def calc_image_hashes(card_pool, save_to=None, hash_size=32, highfreq_factor=4):
    """
    Calculate perceptual hash (pHash) value for each cards in the database, then store them if needed
    :param card_pool: pandas dataframe containing all card information
    :param save_to: path for the pickle file to be saved
    :param hash_size: param for pHash algorithm
    :param highfreq_factor: param for pHash algorithm
    :return: pandas dataframe
    """
    # Since some double-faced cards may result in two different cards, create a new dataframe to store the result
    new_pool = pd.DataFrame(columns=list(card_pool.columns.values))
    new_pool['card_hash'] = np.NaN
    #new_pool['art_hash'] = np.NaN
    for ind, card_info in card_pool.iterrows():
        if ind % 100 == 0:
            print('Calculating hashes: %dth card' % ind)

        card_names = []
        # Double-faced cards have a different json format than normal cards
        if card_info['layout'] in ['transform', 'double_faced_token']:
            if isinstance(card_info['card_faces'], str):
                card_faces = ast.literal_eval(card_info['card_faces'])
            else:
                card_faces = card_info['card_faces']
            for i in range(len(card_faces)):
                card_names.append(card_faces[i]['name'])
        else:  # if card_info['layout'] == 'normal':
            card_names.append(card_info['name'])

        for card_name in card_names:
            # Fetch the image - name can be found based on the card's information
            card_info['name'] = card_name
            img_name = '%s/card_img/png/%s/%s_%s.png' % (Config.data_dir, card_info['set'],
                                                         card_info['collector_number'],
                                                         fetch_data.get_valid_filename(card_info['name']))
            card_img = cv2.imread(img_name)

            # If the image doesn't exist, download it from the URL
            if card_img is None:
                fetch_data.fetch_card_image(card_info,
                                            out_dir='%s/card_img/png/%s' % (Config.data_dir, card_info['set']))
                card_img = cv2.imread(img_name)
            if card_img is None:
                print('WARNING: card %s is not found!' % img_name)

            # Compute value of the card's perceptual hash, then store it to the database
            '''
            img_art = Image.fromarray(card_img[121:580, 63:685])  # For 745*1040 size card image
            art_hash = ih.phash(img_art, hash_size=hash_size, highfreq_factor=highfreq_factor)
            card_info['art_hash'] = art_hash
            '''
            img_card = Image.fromarray(card_img)
            card_hash = ih.phash(img_card, hash_size=hash_size, highfreq_factor=highfreq_factor)
            card_info['card_hash'] = card_hash
            new_pool.loc[0 if new_pool.empty else new_pool.index.max() + 1] = card_info

    # Remove uselesss fields, then pickle it if needed
    new_pool = new_pool[['artist', 'border_color', 'collector_number', 'color_identity', 'colors', 'flavor_text',
                         'image_uris', 'mana_cost', 'legalities', 'name', 'oracle_text', 'rarity', 'type_line',
                         'set', 'set_name', 'power', 'toughness', 'art_hash', 'card_hash']]
    if save_to is not None:
        new_pool.to_pickle(save_to)
    return new_pool


# www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    """
    initialzie a list of coordinates that will be ordered such that the first entry in the list is the top-left,
    the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left
    :param pts: array containing 4 points
    :return: ordered list of 4 points
    """
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    """
    Transform a quadrilateral section of an image into a rectangular area
    From: www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    :param image: source image
    :param pts: 4 corners of the quadrilateral
    :return: rectangular image of the specified area
    """
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    mat = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, mat, (maxWidth, maxHeight))

    # If the image is horizontally long, rotate it by 90
    if maxWidth > maxHeight:
        center = (maxHeight / 2, maxHeight / 2)
        mat_rot = cv2.getRotationMatrix2D(center, 270, 1.0)
        warped = cv2.warpAffine(warped, mat_rot, (maxHeight, maxWidth))

    # return the warped image
    return warped


'''
# The following functions are only used in conjunction with YOLO, and is deprecated:
# - get_outputs_names()
# - post_process()
# - draw_pred() 
# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence using non-maxima suppression
# https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
def post_process(frame, outs, thresh_conf, thresh_nms):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > thresh_conf:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = [ind[0] for ind in cv2.dnn.NMSBoxes(boxes, confidences, thresh_conf, thresh_nms)]
    
    ret = [[class_ids[i], confidences[i], boxes[i]] for i in indices]
    return ret


# Draw the predicted bounding box
def draw_pred(frame, class_id, classes, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (class_id < len(classes))
        label = '%s:%s' % (classes[class_id], label)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
'''


def remove_glare(img):
    """
    Reduce the effect of glaring in the image
    Inspired from:
    http://www.amphident.de/en/blog/preprocessing-for-automatic-pattern-identification-in-wildlife-removing-glare.html
    The idea is to find area that has low saturation but high value, which is what a glare usually look like.
    :param img: source image
    :return: corrected image with glaring smoothened out
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(img_hsv)
    non_sat = (s < 32) * 255  # Find all pixels that are not very saturated

    # Slightly decrease the area of the non-satuared pixels by a erosion operation.
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    non_sat = cv2.erode(non_sat.astype(np.uint8), disk)

    # Set all brightness values, where the pixels are still saturated to 0.
    v[non_sat == 0] = 0
    # filter out very bright pixels.
    glare = (v > 200) * 255

    # Slightly increase the area for each pixel
    glare = cv2.dilate(glare.astype(np.uint8), disk)
    glare_reduced = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 200
    glare = cv2.cvtColor(glare, cv2.COLOR_GRAY2BGR)
    corrected = np.where(glare, glare_reduced, img)
    return corrected


def find_card(img, thresh_c=5, kernel_size=(3, 3), size_thresh=10000):
    """
    Find contours of all cards in the image
    :param img: source image
    :param thresh_c: value of the constant C for adaptive thresholding
    :param kernel_size: dimension of the kernel used for dilation and erosion
    :param size_thresh: threshold for size (in pixel) of the contour to be a candidate
    :return: list of candidate contours
    """
    # Typical pre-processing - grayscale, blurring, thresholding
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, thresh_c)

    # Dilute the image, then erode them to remove minor noises
    kernel = np.ones(kernel_size, np.uint8)
    img_dilate = cv2.dilate(img_thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)

    # Find the contour
    _, cnts, hier = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        #print('no contours')
        return []

    # The hierarchy from cv2.findContours() is similar to a tree: each node has an access to the parent, the first child
    # their previous and next node
    # Using recursive search, find the uppermost contour in the hierarchy that satisfies the condition
    # The candidate contour must be rectangle (has 4 points) and should be larger than a threshold
    cnts_rect = []
    stack = [(0, hier[0][0])]
    while len(stack) > 0:
        i_cnt, h = stack.pop()
        i_next, i_prev, i_child, i_parent = h
        if i_next != -1:
            stack.append((i_next, hier[0][i_next]))
        cnt = cnts[i_cnt]
        size = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if size >= size_thresh and len(approx) == 4:
            cnts_rect.append(approx)
        else:
            if i_child != -1:
                stack.append((i_child, hier[0][i_child]))
    return cnts_rect


def draw_card_graph(exist_cards, card_pool, f_len):
    """
    Given the history of detected cards in the current and several previous frames, draw a simple graph
    displaying the detected cards with its confidence level
    :param exist_cards: History of all detected cards in the previous (f_len) frames
    :param card_pool: pandas dataframe of all card's information
    :param f_len: length of windows (in frames) to consider for confidence level
    :return:
    """
    # Lots of constants to set the dimension of each elements
    w_card = 63  # Width of the card image displayed
    h_card = 88
    gap = 25  # Offset between each elements
    gap_sm = 10  # Small offset
    w_bar = 300  # Length of the confidence bar at 100%
    h_bar = 12
    txt_scale = 0.8
    n_cards_p_col = 4  # Number of cards displayed per one column
    w_img = gap + (w_card + gap + w_bar + gap) * 2  # Dimension of the entire graph (for 2 columns)
    h_img = 480
    img_graph = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    x_anchor = gap
    y_anchor = gap

    i = 0

    # Cards are displayed from the most confident to the least
    # Confidence level is calculated by number of frames that the card was detected in
    for key, val in sorted(exist_cards.items(), key=itemgetter(1), reverse=True)[:n_cards_p_col * 2]:
        card_name = key[:key.find('(') - 1]
        card_set = key[key.find('(') + 1:key.find(')')]
        confidence = sum(val) / f_len
        card_info = card_pool[(card_pool['name'] == card_name) & (card_pool['set'] == card_set)].iloc[0]
        img_name = '%s/card_img/tiny/%s/%s_%s.png' % (Config.data_dir, card_info['set'],
                                                      card_info['collector_number'],
                                                      fetch_data.get_valid_filename(card_info['name']))
        # If the card image is not found, just leave it blank
        if os.path.exists(img_name):
            card_img = cv2.imread(img_name)
        else:
            card_img = np.ones((h_card, w_card))

        # Insert the card image, card name, and confidence bar to the graph
        img_graph[y_anchor:y_anchor + h_card, x_anchor:x_anchor + w_card] = card_img
        cv2.putText(img_graph, '%s (%s)' % (card_name, card_set),
                    (x_anchor + w_card + gap, y_anchor + gap_sm + int(txt_scale * 25)), cv2.FONT_HERSHEY_SIMPLEX,
                    txt_scale, (255, 255, 255), 1)
        cv2.rectangle(img_graph, (x_anchor + w_card + gap, y_anchor + h_card - (gap_sm + h_bar)),
                      (x_anchor + w_card + gap + int(w_bar * confidence), y_anchor + h_card - gap_sm), (0, 255, 0),
                      thickness=cv2.FILLED)
        y_anchor += h_card + gap
        i += 1
        if i % n_cards_p_col == 0:
            x_anchor += w_card + gap + w_bar + gap
            y_anchor = gap
        pass
    return img_graph


def detect_frame(img, card_pool, hash_size=32, highfreq_factor=4, size_thresh=10000,
                 out_path=None, display=True, debug=False):
    """
    Identify all cards in the input frame, display or save the frame if needed
    :param img: input frame
    :param card_pool: pandas dataframe of all card's information
    :param hash_size: param for pHash algorithm
    :param highfreq_factor: param for pHash algorithm
    :param size_thresh: threshold for size (in pixel) of the contour to be a candidate
    :param out_path: path to save the result
    :param display: flag for displaying the result
    :param debug: flag for debug mode
    :return: list of detected card's name/set and resulting image
    """

    img_result = img.copy()  # For displaying and saving
    det_cards = []
    # Detect contours of all cards in the image
    cnts = find_card(img_result, size_thresh=size_thresh)
    for i in range(len(cnts)):
        cnt = cnts[i]
        # For the region of the image covered by the contour, transform them into a rectangular image
        pts = np.float32([p[0] for p in cnt])
        img_warp = four_point_transform(img, pts)

        # To identify the card from the card image, perceptual hashing (pHash) algorithm is used
        # Perceptual hash is a hash string built from features of the input medium. If two media are similar
        # (ie. has similar features), their resulting pHash value will be very close.
        # Using this property, the matching card for the given card image can be found by comparing pHash of
        # all cards in the database, then finding the card that results in the minimal difference in pHash value.
        '''
        img_art = img_warp[47:249, 22:294]
        img_art = Image.fromarray(img_art.astype('uint8'), 'RGB')
        art_hash = ih.phash(img_art, hash_size=hash_size, highfreq_factor=highfreq_factor).hash.flatten()
        card_pool['hash_diff'] = card_pool['art_hash'].apply(lambda x: np.count_nonzero(x != art_hash))
        '''
        img_card = Image.fromarray(img_warp.astype('uint8'), 'RGB')
        # the stored values of hashes in the dataframe is pre-emptively flattened already to minimize computation time
        card_hash = ih.phash(img_card, hash_size=hash_size, highfreq_factor=highfreq_factor).hash.flatten()
        card_pool['hash_diff'] = card_pool['card_hash'].apply(lambda x: np.count_nonzero(x != card_hash))
        min_card = card_pool[card_pool['hash_diff'] == min(card_pool['hash_diff'])].iloc[0]
        card_name = min_card['name']
        card_set = min_card['set']
        det_cards.append((card_name, card_set))
        hash_diff = min_card['hash_diff']

        # Render the result, and display them if needed
        cv2.drawContours(img_result, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(img_result, card_name, (pts[0][0], pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if debug:
            # cv2.rectangle(img_warp, (22, 47), (294, 249), (0, 255, 0), 2)
            cv2.putText(img_warp, card_name + ', ' + str(hash_diff), (0, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('card#%d' % i, img_warp)
    if display:
        cv2.imshow('Result', img_result)
        cv2.waitKey(0)

    if out_path is not None:
        cv2.imwrite(out_path, img_result.astype(np.uint8))
    return det_cards, img_result


def detect_video(capture, card_pool, hash_size=32, highfreq_factor=4, size_thresh=10000,
                 out_path=None, display=True, show_graph=True, debug=False):
    """
    Identify all cards in the continuous video stream, display or save the result if needed
    :param capture: input video stream
    :param card_pool: pandas dataframe of all card's information
    :param hash_size: param for pHash algorithm
    :param highfreq_factor: param for pHash algorithm
    :param size_thresh: threshold for size (in pixel) of the contour to be a candidate
    :param out_path: path to save the result
    :param display: flag for displaying the result
    :param show_graph: flag to show graph
    :param debug: flag for debug mode
    :return: list of detected card's name/set and resulting image
    :return:
    """
    # Get the dimension of the output video, and set it up
    if show_graph:
        img_graph = draw_card_graph({}, pd.DataFrame(), -1)  # Black image of the graph just to get the dimension
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) + img_graph.shape[1]
        height = max(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), img_graph.shape[0])
    else:
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if out_path is not None:
        vid_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (width, height))
    max_num_obj = 0
    f_len = 10  # number of frames to consider to check for existing cards
    exist_cards = {}
    try:
        while True:
            ret, frame = capture.read()
            start_time = time.time()
            if not ret:
                # End of video
                print("End of video. Press any key to exit")
                cv2.waitKey(0)
                break
            # Detect all cards from the current frame
            det_cards, img_result = detect_frame(frame, card_pool, hash_size=hash_size, highfreq_factor=highfreq_factor,
                                                 size_thresh=size_thresh, out_path=None, display=False, debug=debug)
            if show_graph:
                # If the card was already detected in the previous frame, append 1 to the list
                # If the card previously detected was not found in this trame, append 0 to the list
                # If the card wasn't previously detected, make a new list and add 1 to it
                # If the same card is detected multiple times in the same frame, keep track of the duplicates
                # The confidence will be calculated based on the number of frames the card was detected for
                det_cards_count = collections.Counter(det_cards).items()
                det_cards_list = []
                for card, count in det_cards_count:
                    card_name, card_set = card
                    for i in range(count): 1
                    key = '%s (%s) #%d' % (card_name, card_set, i + 1)
                    det_cards_list.append(key)
                gone = []
                for key, val in exist_cards.items():
                    if key in det_cards_list:
                        exist_cards[key] = exist_cards[key][1 - f_len:] + [1]
                    else:
                        exist_cards[key] = exist_cards[key][1 - f_len:] + [0]
                    if len(val) == f_len and sum(val) == 0:
                        gone.append(key)
                for key in det_cards_list:
                    if key not in exist_cards.keys():
                        exist_cards[key] = [1]
                for key in gone:
                    exist_cards.pop(key)

                # Draw the graph based on the history of detected cards, then concatenate it with the result image
                img_graph = draw_card_graph(exist_cards, card_pool, f_len)
                img_save = np.zeros((height, width, 3), dtype=np.uint8)
                img_save[0:img_result.shape[0], 0:img_result.shape[1]] = img_result
                img_save[0:img_graph.shape[0], img_result.shape[1]:img_result.shape[1] + img_graph.shape[1]] = img_graph
            else:
                img_save = img_result

            # Display the result
            if display:
                cv2.imshow('result', img_save)
            if debug:
                max_num_obj = max(max_num_obj, len(det_cards))
                for i in range(len(det_cards), max_num_obj):
                    cv2.imshow('card#%d' % i, np.zeros((1, 1), dtype=np.uint8))

            elapsed_ms = (time.time() - start_time) * 1000
            print('Elapsed time: %.2f ms' % elapsed_ms)
            if out_path is not None:
                vid_writer.write(img_save.astype(np.uint8))
            cv2.waitKey(1)
    except KeyboardInterrupt:
        capture.release()
        if out_path is not None:
            vid_writer.release()
        cv2.destroyAllWindows()


def main():
    # Specify paths for all necessary files
    #test_path = os.path.abspath('test_file/test4.mp4')
    test_path = None
    out_dir = 'out'
    hash_size = 32
    highfreq_factor = 4

    pck_path = os.path.abspath('card_pool_%d_%d.pck' % (hash_size, highfreq_factor))
    if os.path.isfile(pck_path):
        card_pool = pd.read_pickle(pck_path)
    else:
        # Merge database for all cards, then calculate pHash values of each, store them
        df_list = []
        for set_name in Config.all_set_list:
            csv_name = '%s/csv/%s.csv' % (Config.data_dir, set_name)
            df = fetch_data.load_all_cards_text(csv_name)
            df_list.append(df)
        card_pool = pd.concat(df_list, sort=True)
        card_pool.reset_index(drop=True, inplace=True)
        card_pool.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')

        card_pool = calc_image_hashes(card_pool, save_to=pck_path, hash_size=hash_size, highfreq_factor=highfreq_factor)
    card_pool = card_pool[['name', 'set', 'collector_number', 'card_hash']]

    # ImageHash is basically just one numpy.ndarray with (hash_size)^2 number of bits. pre-emptively flattening it
    # significantly increases speed for subtracting hashes in the future.
    card_pool['card_hash'] = card_pool['card_hash'].apply(lambda x: x.hash.flatten())


    # If the test file isn't given, use webcam to capture video
    if test_path is None:
        capture = cv2.VideoCapture(0)
        detect_video(capture, card_pool, out_path='%s/result.avi' % out_dir, display=True, show_graph=True, debug=False)
        capture.release()
    else:
        # Save the detection result if out_dir is provided
        if out_dir is None or out_dir == '':
            out_path = None
        else:
            f_name = os.path.split(test_path)[1]
            out_path = '%s/%s.avi' % (out_dir, f_name[:f_name.find('.')])

        if not os.path.isfile(test_path):
            print('The test file %s doesn\'t exist!' % os.path.abspath(test_path))
            return
        # Check if test file is image or video
        test_ext = test_path[test_path.find('.') + 1:]
        if test_ext in ['jpg', 'jpeg', 'bmp', 'png', 'tiff']:
            # Test file is an image
            img = cv2.imread(test_path)
            detect_frame(img, card_pool, out_path=out_path)
        else:
            # Test file is a video
            capture = cv2.VideoCapture(test_path)
            detect_video(capture, card_pool, out_path=out_path, display=True, show_graph=True, debug=False)
            capture.release()
    pass


if __name__ == '__main__':
    main()
