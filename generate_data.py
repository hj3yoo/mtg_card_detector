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
    # Mana symbol - They are located on the top right side of the card, next to the name.
    # Their position is stationary, and is right-aligned.
    has_mana_cost = isinstance(card_info['mana_cost'], str)  # Cards with no mana cost will have nan
    is_planeswalker = 'Planeswalker' in card_info['type_line']
    if has_mana_cost:
        mana_cost = re.findall('\{(.*?)\}', card_info['mana_cost'])
        x2 = 683
        y1 = 67

        # Cards with specific type or from old sets have their symbol at a different position
        if is_planeswalker:
            y1 -= 17
        if card_info['set'] in ['8ed', 'mrd', 'dst', '5dn']:
            y1 -= 2

        for i in reversed(range(len(mana_cost))):
            # Hybrid mana symbol are larger than a normal symbol
            is_hybrid = '/' in mana_cost[i]
            if is_hybrid:
                box = [(x2 - 47, y1 - 8), (x2 + 2, y1 + 43)]  # (x1, y1), (x2, y2)
                x2 -= 45
            else:
                box = [(x2 - 39, y1), (x2, y1 + 41)]  # (x1, y1), (x2, y2)
                x2 -= 37
            img_symbol = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            if display:
                cv2.imshow('symbol', img_symbol)
                cv2.waitKey(0)


def main():
    #bg_images = load_dtd()
    #bg = Backgrounds()
    #bg.get_random(display=True)
    df = fetch_data.load_all_cards_text('data/csv/dgm.csv')
    #repeat = 'y'
    while True:
        card_info = df.iloc[random.randint(0, df.shape[0] - 1)]
        print(card_info['name'])
        card_img = cv2.imread('data/png/%s/%s_%s.png' % (card_info['set'], card_info['collector_number'],
                                                         fetch_data.get_valid_filename(card_info['name'])))
        if card_img is None:
            fetch_data.fetch_card_image(card_info)
            card_img = cv2.imread('data/png/%s/%s_%s.png' % (card_info['set'], card_info['collector_number'],
                                                             fetch_data.get_valid_filename(card_info['name'])))
        sys.stdout.flush()
        apply_bounding_box(card_img, card_info, display=True)
        #repeat = input('y to repeat, n to finish')
    return


if __name__ == '__main__':
    main()
