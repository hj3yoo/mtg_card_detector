import cv2
import numpy as np
import pandas as pd
import imagehash as ih
import os
import ast
import queue
import sys
import math
import random
import collections
from operator import itemgetter
import time
from PIL import Image
import fetch_data
import transform_data

card_width = 315
card_height = 440


def calc_image_hashes(card_pool, save_to=None, hash_size=32, highfreq_factor=4):
    new_pool = pd.DataFrame(columns=list(card_pool.columns.values))
    new_pool['card_hash'] = np.NaN
    new_pool['art_hash'] = np.NaN
    for ind, card_info in card_pool.iterrows():
        if ind % 100 == 0:
            print(ind)

        card_names = []
        if card_info['layout'] in ['transform', 'double_faced_token']:
            if isinstance(card_info['card_faces'], str):  # For some reason, dict isn't being parsed in the previous step
                card_faces = ast.literal_eval(card_info['card_faces'])
            else:
                card_faces = card_info['card_faces']
            for i in range(len(card_faces)):
                card_names.append(card_faces[i]['name'])
        else:  # if card_info['layout'] == 'normal':
            card_names.append(card_info['name'])

        for card_name in card_names:
            card_info['name'] = card_name
            img_name = '%s/card_img/png/%s/%s_%s.png' % (transform_data.data_dir, card_info['set'],
                                                         card_info['collector_number'],
                                                         fetch_data.get_valid_filename(card_info['name']))
            card_img = cv2.imread(img_name)
            if card_img is None:
                fetch_data.fetch_card_image(card_info,
                                            out_dir='%s/card_img/png/%s' % (transform_data.data_dir, card_info['set']))
                card_img = cv2.imread(img_name)
            if card_img is None:
                print('WARNING: card %s is not found!' % img_name)
            #img_art = Image.fromarray(card_img[121:580, 63:685])  # For 745*1040 size card image
            #art_hash = ih.phash(img_art, hash_size=32, highfreq_factor=4)
            #card_pool.at[ind, 'art_hash'] = art_hash
            img_card = Image.fromarray(card_img)
            card_hash = ih.phash(img_card, hash_size=hash_size, highfreq_factor=highfreq_factor)
            #card_pool.at[ind, 'card_hash'] = card_hash
            card_info['card_hash'] = card_hash
            #print(new_pool.index.max())
            new_pool.loc[0 if new_pool.empty else new_pool.index.max() + 1] = card_info

    new_pool = new_pool[['artist', 'border_color', 'collector_number', 'color_identity', 'colors', 'flavor_text',
                         'image_uris', 'mana_cost', 'legalities', 'name', 'oracle_text', 'rarity', 'type_line',
                         'set', 'set_name', 'power', 'toughness', 'art_hash', 'card_hash']]
    if save_to is not None:
        new_pool.to_pickle(save_to)
    return new_pool


# www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
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


# www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def four_point_transform(image, pts):
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


def remove_glare(img):
    """
    Inspired from:
    http://www.amphident.de/en/blog/preprocessing-for-automatic-pattern-identification-in-wildlife-removing-glare.html
    The idea is to find area that has low saturation but high value, which is what a glare usually look like.
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


def find_card(img, thresh_c=5, kernel_size=(3, 3), size_thresh=5000):
    # Typical pre-processing - grayscale, blurring, thresholding
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, thresh_c)

    # Dilute the image, then erode them to remove minor noises
    kernel = np.ones(kernel_size, np.uint8)
    img_dilate = cv2.dilate(img_thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)

    # Find the contour
    #img_contour = img_erode.copy()
    _, cnts, hier = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        print('no contours')
        return []
    cv2.drawContours(img, cnts, -1, (0, 0, 255), 1)
    '''
    next = 0
    while next != -1:
        img_copy = img.copy()
        print(hier[0][next])
        cv2.drawContours(img_copy, cnts[hier[0][next][0]], -1, (0, 255, 0), 2)
        cv2.imshow('hi', img_copy)
        cv2.waitKey(0)
        next = hier[0][next][0]
    '''
    #img_contour = cv2.cvtColor(img_contour, cv2.COLOR_GRAY2BGR)
    #img_contour = cv2.drawContours(img_contour, cnts, -1, (0, 255, 0), 1)
    #cv2.imshow('test', img_contour)

    '''
    The hierarchy from cv2.findContours() is similar to a tree: each node has an access to the parent, the first child, 
    their previous and next node 
    Using (preorder) depth-first search, find the uppermost contour in the hierarchy that satisfies the condition
    The candidate contour must be rectangle (has 4 points) and should be larger than a threshold
    '''

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
        if size >= size_thresh:
            cv2.drawContours(img, [cnt], -1, (255, 0, 0), 1)
            #print(size)
            if len(approx) == 4:
                cnts_rect.append(approx)
        else:
            if i_child != -1:
                stack.append((i_child, hier[0][i_child]))


    '''
    # For each contours detected, check if they are large enough and are rectangle
    ind_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)
    for i in range(len(cnts)):
        peri = cv2.arcLength(cnts[ind_sort[i]], True)
        approx = cv2.approxPolyDP(cnts[ind_sort[i]], 0.04 * peri, True)
        if len(approx) == 4:
            cnts_rect.append(approx)
    '''

    return cnts_rect


def draw_card_graph(exist_cards, card_pool, f_len):
    w_card = 63
    h_card = 88
    gap = 25
    gap_sm = 10
    w_bar = 300
    h_bar = 12
    txt_scale = 0.8
    n_cards_p_col = 4
    w_img = gap + (w_card + gap + w_bar + gap) * 2
    #h_img = gap + (h_card + gap) * n_cards_p_col
    h_img = 480
    img_graph = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    x_anchor = gap
    y_anchor = gap

    i = 0
    for key, val in sorted(exist_cards.items(), key=itemgetter(1), reverse=True)[:n_cards_p_col * 2]:
        card_name = key[:key.find('(') - 1]
        card_set = key[key.find('(') + 1:key.find(')')]
        confidence = sum(val) / f_len
        card_info = card_pool[(card_pool['name'] == card_name) & (card_pool['set'] == card_set)].iloc[0]
        img_name = '%s/card_img/tiny/%s/%s_%s.png' % (transform_data.data_dir, card_info['set'],
                                                     card_info['collector_number'],
                                                     fetch_data.get_valid_filename(card_info['name']))
        card_img = cv2.imread(img_name)
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


def detect_frame(net, classes, img, card_pool, thresh_conf=0.5, thresh_nms=0.4, in_dim=(416, 416), card_size=1000,
                 out_path=None, display=True, debug=False):
    start_1 = time.time()
    elapsed = []
    '''
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(img, 1 / 255, in_dim, [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))
    elapsed.append((time.time() - start_1) * 1000)

    start_2 = time.time()
    img_result = img.copy()

    # Remove the bounding boxes with low confidence
    obj_list = post_process(img, outs, thresh_conf, thresh_nms)
    for obj in obj_list:
        class_id, confidence, box = obj
        left, top, width, height = box
        draw_pred(img_result, class_id, classes, confidence, left, top, left + width, top + height)
    elapsed.append((time.time() - start_2) * 1000)
    '''
    img_result = img.copy()
    # Put efficiency information. The function getPerfProfile returns the
    # overall time for inference(t) and the timings for each of the layers(in layersTimes)
    #if display:
    #    t, _ = net.getPerfProfile()
    #    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    #    cv2.putText(img_result, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    '''
    Assuming that the model has properly identified all cards, there should be 1 card that can be classified per
    bounding box. Find the largest rectangular contour from the region of interest, and identify the card by  
    comparing the perceptual hashing of the image with the other cards' image from the database.
    '''
    det_cards = []
    start_3 = time.time()
    cnts = find_card(img_result)
    for i in range(len(cnts)):
        cnt = cnts[i]
        # ignore any contours smaller than threshold
        elapsed.append((time.time() - start_3) * 1000)
        start_4 = time.time()
        pts = np.float32([p[0] for p in cnt])
        img_warp = four_point_transform(img, pts)
        img_warp = cv2.resize(img_warp, (card_width, card_height))
        elapsed.append((time.time() - start_4) * 1000)
        '''
        img_art = img_warp[47:249, 22:294]
        img_art = Image.fromarray(img_art.astype('uint8'), 'RGB')
        art_hash = ih.phash(img_art, hash_size=32, highfreq_factor=4)
        card_pool['hash_diff'] = card_pool['art_hash'] - art_hash
        min_cards = card_pool[card_pool['hash_diff'] == min(card_pool['hash_diff'])]
        card_name = min_cards.iloc[0]['name']
        '''
        start_5 = time.time()
        img_card = Image.fromarray(img_warp.astype('uint8'), 'RGB')
        card_hash = ih.phash(img_card, hash_size=32, highfreq_factor=4).hash.flatten()
        card_pool['hash_diff'] = card_pool['card_hash'].apply(lambda x: np.count_nonzero(x != card_hash))
        min_cards = card_pool[card_pool['hash_diff'] == min(card_pool['hash_diff'])]
        card_name = min_cards.iloc[0]['name']
        card_set = min_cards.iloc[0]['set']
        det_cards.append((card_name, card_set))
        hash_diff = min_cards.iloc[0]['hash_diff']
        elapsed.append((time.time() - start_5) * 1000)

        # Display the result
        if debug:
            # cv2.rectangle(img_warp, (22, 47), (294, 249), (0, 255, 0), 2)
            cv2.putText(img_warp, card_name + ', ' + str(hash_diff), (0, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.drawContours(img_result, [cnt], -1, (0, 255, 0), 1)
        cv2.putText(img_result, card_name, (pts[0][0], pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if debug:
            cv2.imshow('card#%d' % i, img_warp)
        #if debug:
        #    cv2.imshow('card#%d' % i, np.zeros((1, 1), dtype=np.uint8))

    if out_path is not None:
        cv2.imwrite(out_path, img_result.astype(np.uint8))
    elapsed = [(time.time() - start_1) * 1000] + elapsed
    #print(', '.join(['%.2f' % t for t in elapsed]))
    return det_cards, img_result


def detect_video(net, classes, capture, card_pool, thresh_conf=0.5, thresh_nms=0.4, in_dim=(416, 416), out_path=None,
                 display=True, debug=False):
    if out_path is not None:
        img_graph = draw_card_graph({}, None, -1)  # Black image of the graph just to get the dimension
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) + img_graph.shape[1]
        height = max(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), img_graph.shape[0])
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
            # Use the YOLO model to identify each cards annonymously
            start_yolo = time.time()
            det_cards, img_result = detect_frame(net, classes, frame, card_pool, thresh_conf=thresh_conf,
                                                 thresh_nms=thresh_nms, in_dim=in_dim, out_path=None, display=display,
                                                 debug=debug)
            elapsed_yolo = (time.time() - start_yolo) * 1000
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
            start_graph = time.time()
            img_graph = draw_card_graph(exist_cards, card_pool, f_len)
            elapsed_graph = (time.time() - start_graph) * 1000
            #if debug:
            #    max_num_obj = max(max_num_obj, len(obj_list))
            #    for i in range(len(obj_list), max_num_obj):
            #        cv2.imshow('card#%d' % i, np.zeros((1, 1), dtype=np.uint8))

            start_display = time.time()
            img_save = np.zeros((height, width, 3), dtype=np.uint8)
            img_save[0:img_result.shape[0], 0:img_result.shape[1]] = img_result
            img_save[0:img_graph.shape[0], img_result.shape[1]:img_result.shape[1] + img_graph.shape[1]] = img_graph
            if display:
                cv2.imshow('result', img_save)
            elapsed_display = (time.time() - start_display) * 1000

            elapsed_ms = (time.time() - start_time) * 1000
            print('Elapsed time: %.2f ms, %.2f, %.2f, %.2f' % (elapsed_ms, elapsed_yolo, elapsed_graph, elapsed_display))
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
    test_path = os.path.abspath('test_file/test4.mp4')
    #weight_path = 'backup/tiny_yolo_10_39500.weights'
    #cfg_path = 'cfg/tiny_yolo_10.cfg'
    #class_path = "data/obj_10.names"
    weight_path = 'weights/second_general/tiny_yolo_final.weights'
    cfg_path = 'cfg/tiny_yolo_old.cfg'
    class_path = 'data/obj.names'
    out_dir = 'out'
    if not os.path.isfile(test_path):
        print('The test file %s doesn\'t exist!' % os.path.abspath(test_path))
        return
    if not os.path.isfile(weight_path):
        print('The weight file %s doesn\'t exist!' % os.path.abspath(test_path))
        return
    if not os.path.isfile(cfg_path):
        print('The config file %s doesn\'t exist!' % os.path.abspath(test_path))
        return
    if not os.path.isfile(class_path):
        print('The class file %s doesn\'t exist!' % os.path.abspath(test_path))
        return


    '''
    df_list = []
    for set_name in fetch_data.all_set_list:
        csv_name = '%s/csv/%s.csv' % (transform_data.data_dir, set_name)
        df = fetch_data.load_all_cards_text(csv_name)
        df_list.append(df)
        #print(df)
    card_pool = pd.concat(df_list, sort=True)
    card_pool.reset_index(drop=True, inplace=True)
    card_pool.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')
    for hash_size in [8, 16, 32, 64]:
        for highfreq_factor in [4, 8, 16, 32]:
            pck_name = 'card_pool_%d_%d.pck' % (hash_size, highfreq_factor)
            if not os.path.exists(pck_name):
                print(pck_name)
                calc_image_hashes(card_pool, save_to=pck_name, hash_size=hash_size, highfreq_factor=highfreq_factor)
    '''
    #csv_name = '%s/csv/%s.csv' % (transform_data.data_dir, 'rtr')
    #card_pool = fetch_data.load_all_cards_text(csv_name)
    #card_pool = calc_image_hashes(card_pool, save_to='card_pool.pck')
    #return
    card_pool = pd.read_pickle('card_pool_32_4.pck')
    #card_pool = card_pool[(card_pool['set'] == 'rtr') | (card_pool['set'] == 'isd')]
    card_pool = card_pool[['name', 'set', 'collector_number', 'card_hash']]

    # ImageHash is basically just one numpy.ndarray with (hash_size)^2 number of bits. pre-emptively flattening it
    # significantly increases speed for subtracting hashes in the future.
    card_pool['card_hash'] = card_pool['card_hash'].apply(lambda x: x.hash.flatten())

    thresh_conf = 0.01
    thresh_nms = 0.8

    # Setup
    # Read class names from text file
    with open(class_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    # Load up the neural net using the config and weights
    net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Save the detection result if out_dir is provided
    if out_dir is None or out_dir == '':
        out_path = None
    else:
        f_name = os.path.split(test_path)[1]
        out_path = out_dir + '/' + f_name[:f_name.find('.')] + '.avi'
    # Check if test file is image or video
    test_ext = test_path[test_path.find('.') + 1:]

    if test_ext in ['jpg', 'jpeg', 'bmp', 'png', 'tiff']:
        img = cv2.imread(test_path)
        detect_frame(net, classes, img, card_pool, out_path=out_path, thresh_conf=thresh_conf, thresh_nms=thresh_nms)
    else:
        capture = cv2.VideoCapture(0)
        detect_video(net, classes, capture, card_pool, out_path=out_path, thresh_conf=thresh_conf, thresh_nms=thresh_nms,
                     display=True, debug=False)
        capture.release()
    pass


if __name__ == '__main__':
    main()
