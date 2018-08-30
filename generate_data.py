from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import pickle
import math
import random
import os
import re
import cv2
import fetch_data
import sys
import numpy as np
import pandas as pd
from transform_data import ExtractedObject

# Referenced from geaxgx's playing-card-detection: https://github.com/geaxgx/playing-card-detection
class Backgrounds:
    def __init__(self, images=None, dumps_dir='data/dtd/images'):
        if images is not None:
            self._images = images
        else:  # load from pickle
            if not os.path.exists(dumps_dir):
                print('Warning: directory for dump %s doesn\'t exist' % dumps_dir)
                return
            self._images = []
            for dump_name in glob(dumps_dir + '/*.pck'):
                with open(dump_name, 'rb') as dump:
                    print('Loading ' + dump_name)
                    images = pickle.load(dump)
                    self._images += images
            if len(self._images) == 0:
                self._images = load_dtd()
        print('# of images loaded: %d' % len(self._images))

    def get_random(self, display=False):
        bg = self._images[random.randint(0, len(self._images) - 1)]
        if display:
            plt.show(bg)
        return bg


def load_dtd(dtd_dir='data/dtd/images', dump_it=True, dump_batch_size=1000):
    if not os.path.exists(dtd_dir):
        print('Warning: directory for DTD 5s doesn\'t exist.' % dtd_dir)
        print('You can download the dataset using this command:'
              '!wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz')
        return []
    bg_images = []
    # Search the directory for all images, and append them
    for subdir in glob(dtd_dir + "/*"):
        for f in glob(subdir + "/*.jpg"):
            bg_images.append(mpimage.imread(f))
    print("# of images loaded :", len(bg_images))

    # Save them as a pickle if necessary
    if dump_it:
        for i in range(math.ceil(len(bg_images) / dump_batch_size)):
            dump_name = '%s/dtd_dump_%d.pck' % (dtd_dir, i)
            with open(dump_name, 'wb') as dump:
                print('Dumping ' + dump_name)
                pickle.dump(bg_images[i * dump_batch_size:(i + 1) * dump_batch_size], dump)

    return bg_images


def apply_bounding_box(img, card_info, display=False):
    # List of detected objects to be fed into the neural net
    # The first object is the entire card
    detected_object_list = [ExtractedObject('card', [(0, 0), (len(img[0]), 0), (len(img[0]), len(img)), (0, len(img))])]
    # Mana symbol - They are located on the top right side of the card, next to the name
    # Their position is stationary, and is right-aligned.
    has_mana_cost = isinstance(card_info['mana_cost'], str)  # Cards with no mana cost will have nan
    if has_mana_cost:
        mana_cost = re.findall('\{(.*?)\}', card_info['mana_cost'])
        x_anchor = 683
        y_anchor = 65

        # Cards with specific type or from old sets have their symbol at a different position
        if card_info['set'] in ['8ed', 'mrd', 'dst', '5dn']:
            y_anchor -= 2

        for i in reversed(range(len(mana_cost))):
            # Hybrid mana symbol are larger than a normal symbol
            is_hybrid = '/' in mana_cost[i]
            if is_hybrid:
                x1 = x_anchor - 47
                x2 = x_anchor + 2
                y1 = y_anchor - 8
                y2 = y_anchor + 43
                x_anchor -= 45
            else:
                x1 = x_anchor - 39
                x2 = x_anchor
                y1 = y_anchor
                y2 = y_anchor + 43
                x_anchor -= 37
            # Append them to the list of bounding box with the appropriate label
            symbol_name = 'mana_symbol:' + mana_cost[i]
            key_pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            detected_object_list.append(ExtractedObject(symbol_name, key_pts))

            if display:
                img_symbol = img[y1:y2, x1:x2]
                cv2.imshow('symbol', img_symbol)
                cv2.waitKey(0)

    # Set symbol - located on the right side of the type box in the centre of the card, next to the card type
    # Only one symbol exists, and its colour varies by rarity.
    if card_info['set'] in ['8ed']:
        x1 = 622
        x2 = 670
    elif card_info['set'] in ['mrd', 'm10', 'm11', 'm12', 'm13', 'm14']:
        x1 = 602
        x2 = 684
    elif card_info['set'] in ['dst']:
        x1 = 636
        x2 = 673
    elif card_info['set'] in ['5dn']:
        x1 = 630
        x2 = 675
    elif card_info['set'] in ['bok', 'rtr']:
        x1 = 633
        x2 = 683
    elif card_info['set'] in ['sok', 'mbs']:
        x1 = 638
        x2 = 683
    elif card_info['set'] in ['rav']:
        x1 = 640
        x2 = 678
    elif card_info['set'] in ['csp']:
        x1 = 650
        x2 = 683
    elif card_info['set'] in ['tsp', 'lrw', 'zen', 'wwk', 'ths']:
        x1 = 640
        x2 = 683
    elif card_info['set'] in ['plc', 'fut', 'shm', 'eve']:
        x1 = 625
        x2 = 685
    elif card_info['set'] in ['10e']:
        x1 = 623
        x2 = 680
    elif card_info['set'] in ['mor', 'roe', 'bng']:
        x1 = 637
        x2 = 687
    elif card_info['set'] in ['ala', 'arb']:
        x1 = 635
        x2 = 680
    elif card_info['set'] in ['nph']:
        x1 = 642
        x2 = 678
    elif card_info['set'] in ['gtc']:
        x1 = 610
        x2 = 683
    elif card_info['set'] in ['dgm']:
        x1 = 618
        x2 = 678
    else:
        x1 = 630
        x2 = 683
    y1 = 589
    y2 = 636
    # Append them to the list of bounding box with the appropriate label
    symbol_name = 'set_symbol:' + card_info['set']
    key_pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    detected_object_list.append(ExtractedObject(symbol_name, key_pts))

    if display:
        img_symbol = img[y1:y2, x1:x2]
        cv2.imshow('symbol', img_symbol)
        cv2.waitKey(0)

    # Name box - The long bar on the top with card name and mana symbols
    # TODO

    # Type box - The long bar on the middle with card type and set symbols
    # TODO

    # Image box - the large image on the top half of the card
    # TODO
    return detected_object_list


def main():
    random.seed()
    #bg_images = load_dtd()
    #bg = Backgrounds()
    #bg.get_random(display=True)

    card_pool = pd.DataFrame()
    for set_name in fetch_data.all_set_list:
        df = fetch_data.load_all_cards_text('data/csv/%s.csv' % set_name)
        for _ in range(3):
            card_info = df.iloc[random.randint(0, df.shape[0] - 1)]
            # Currently ignoring planeswalker cards due to their different card layout
            is_planeswalker = 'Planeswalker' in card_info['type_line']
            if not is_planeswalker:
                card_pool = card_pool.append(card_info)

    for _, card_info in card_pool.iterrows():
        img_name = '../usb/data/png/%s/%s_%s.png' % (card_info['set'], card_info['collector_number'],
                                                     fetch_data.get_valid_filename(card_info['name']))
        print(img_name)
        card_img = cv2.imread(img_name)
        if card_img is None:
            fetch_data.fetch_card_image(card_info, out_dir='../usb/data/png/%s' % card_info['set'])
            card_img = cv2.imread(img_name)
        detected_object_list = apply_bounding_box(card_img, card_info, display=True)
        print(detected_object_list)
    return


if __name__ == '__main__':
    main()
