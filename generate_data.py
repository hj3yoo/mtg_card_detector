from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import pickle
import math
import random
import os
import cv2
import fetch_data
import numpy as np
import pandas as pd
import transform_data


class Backgrounds:
    """
    Container class for all background images for generator
    Referenced from geaxgx's playing-card-detection: https://github.com/geaxgx/playing-card-detection
    """
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
    """
    Load Describable Texture Dataset (DTD) from local
    :param dtd_dir: path of the DTD images folder
    :param dump_it: flag for pickling it
    :param dump_batch_size: # of images stored per pickle file
    :return: list of all DTD images
    """
    if not os.path.exists(dtd_dir):
        print('Warning: directory for DTD %s doesn\'t exist.' % dtd_dir)
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
    """
    Given a card image, extract specific features that can be used to train a model.
    Note: Mana & set symbols are deprecated from the feature list. Refer to previous commits for their implementation:
    https://github.com/hj3yoo/mtg_card_detector/tree/bb34d4e13da0f4753fbdefee837f54b16149d3ef
    :param img: image of the card
    :param card_info: characteristics of this card
    :param display: flag for displaying the extracted features
    :return:
    """
    # List of detected objects to be fed into the neural net
    # The first object is the entire card
    detected_object_list = [transform_data.ExtractedObject('card', [(0, 0), (len(img[0]), 0), (len(img[0]), len(img)),
                                                                    (0, len(img))])]
    return detected_object_list


def main():
    random.seed()
    #bg_images = load_dtd()
    #bg = Backgrounds()
    #bg.get_random(display=True)

    card_pool = pd.DataFrame()
    for set_name in fetch_data.all_set_list:
        df = fetch_data.load_all_cards_text('data/csv/%s.csv' % set_name)
        #for _ in range(3):
        #    card_info = df.iloc[random.randint(0, df.shape[0] - 1)]
        #    # Currently ignoring planeswalker cards due to their different card layout
        #    is_planeswalker = 'Planeswalker' in card_info['type_line']
        #    if not is_planeswalker:
        #        card_pool = card_pool.append(card_info)
        card_pool = card_pool.append(df)
    '''
    print(card_pool)
    mana_symbol_set = set()
    for _, card_info in card_pool.iterrows():
        has_mana_cost = isinstance(card_info['mana_cost'], str)
        if has_mana_cost:
            mana_cost = re.findall('\{(.*?)\}', card_info['mana_cost'])
            for symbol in mana_cost:
                mana_symbol_set.add(symbol)

    print(mana_symbol_set)
    '''

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
