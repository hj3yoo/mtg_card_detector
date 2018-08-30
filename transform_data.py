import random
import math
import cv2
import numpy as np
import imutils
import pandas as pd
import fetch_data

card_mask = cv2.imread('data/mask.png')


class ImageGenerator:
    """
    A template for generating a training image.
    """
    def __init__(self, img_bg, cards, width, height):
        """
        :param img_bg: background (textile) image
        :param cards: list of Card objects
        :param width: width of the training image
        :param height: height of the training image
        """
        self.img_bg = img_bg
        self.cards = cards
        self.img_result = None
        self.width = width
        self.height = height
        pass

    def display(self):
        """
        Display the current state of the generator
        :return: none
        """
        img_bg = cv2.resize(self.img_bg, (self.width, self.height))

        for card in self.cards:
            card_x = int(card.x + 0.5)
            card_y = int(card.y + 0.5)
            print(card_x, card_y, card.theta, card.scale)
            # Scale & rotate card image
            img_card = cv2.resize(card.img, (int(len(card.img[0]) * card.scale), int(len(card.img) * card.scale)))
            mask_scale = cv2.resize(card_mask, (int(len(card_mask[0]) * card.scale), int(len(card_mask) * card.scale)))
            img_mask = cv2.bitwise_and(img_card, mask_scale)
            img_rotate = imutils.rotate_bound(img_mask, card.theta / math.pi * 180)

            # Calculate the position of the card image in relation to the background
            # Crop the card image if it's out of boundary
            card_w = len(img_rotate[0])
            card_h = len(img_rotate)
            card_crop_x1 = max(0, card_w // 2 - card_x)
            card_crop_x2 = min(card_w, card_w // 2 + len(img_bg[0]) - card_x)
            card_crop_y1 = max(0, card_h // 2 - card_y)
            card_crop_y2 = min(card_h, card_h // 2 + len(img_bg) - card_y)
            img_card_crop = img_rotate[card_crop_y1:card_crop_y2, card_crop_x1:card_crop_x2]

            # Calculate the position of the corresponding area in the background
            bg_crop_x1 = max(0, card_x - (card_w // 2))
            bg_crop_x2 = min(len(img_bg[0]), int(card_x + (card_w / 2) + 0.5))
            bg_crop_y1 = max(0, card_y - (card_h // 2))
            bg_crop_y2 = min(len(img_bg), int(card_y + (card_h / 2) + 0.5))
            img_bg_crop = img_bg[bg_crop_y1:bg_crop_y2, bg_crop_x1:bg_crop_x2]

            # Override the background with the current card
            img_bg_crop = np.where(img_card_crop, img_card_crop, img_bg_crop)
            img_bg[bg_crop_y1:bg_crop_y2, bg_crop_x1:bg_crop_x2] = img_bg_crop
        cv2.imshow('Result', img_bg)
        cv2.waitKey(0)
        pass

    def generate_horizontal_span(self):
        """
        Generating the first scenario where the cards are laid out in a straight horizontal line
        :return: none
        """
        pass

    def generate_vertical_span(self):
        """
        Generating the second scenario where the cards are laid out in a straight vertical line
        :return: none
        """
        pass

    def generate_fan_out(self):
        """
        Generating the third scenario where the cards are laid out in a fan shape
        :return: none
        """
        pass

    def export_training_data(self, out_dir):
        """
        Export the generated training image along with the txt file for all bounding boxes
        :return: none
        """
        pass


class Card:
    """
    A class for storing required information about a card in relation to the ImageGenerator
    """
    def __init__(self, img, card_info, objects, generator=None, x=None, y=None, theta=None, scale=None):
        """
        :param img: image of the card
        :param card_info: details like name, mana cost, type, set, etc
        :param objects: list of ExtractedObjects like mana & set symbol, etc
        :param generator: ImageGenerator object that the card is bound to
        :param x: X-coordinate of the card's centre in relation to the generator
        :param y: Y-coordinate of the card's centre in relation to the generator
        :param theta: angle of rotation of the card in relation to the generator
        :param scale: scale of the card in the generator in relation to the original image
        """
        self.img = img
        self.info = card_info
        self.objects = objects
        self.generator = generator
        self.x = x
        self.y = y
        self.theta = theta
        self.scale = scale
        pass

    def bind_to_generator(self, generator, x=0, y=0, theta=0.0, scale=1.0):
        """
        Bind this card to an ImageGenerator object.
        :param generator: generator to be bound with
        :param x: new X-coordinate for the centre of the card
        :param y: new Y-coordinate for the centre of the card
        :param theta: new angle for the card
        :param scale: new scale for the card
        :return: none
        """
        self.generator = generator
        self.x = x
        self.y = y
        self.theta = theta
        self.scale = scale
        generator.cards.append(self)
        pass

    def shift(self, x, y):
        """
        Apply a X/Y translation on this image
        :param x: amount of X-translation. If range is given, translate by a random amount within that range
        :param y: amount of Y-translation. Refer to x when a range is given.
        :return: none
        """
        if isinstance(x, tuple) or (isinstance(x, list) and len(x) == 2):
            self.x += random.uniform(x[0], x[1])
        else:
            self.x += x
        if isinstance(y, tuple) or (isinstance(y, list) and len(y) == 2):
            self.y += random.uniform(y[0], y[1])
        else:
            self.y += y
        pass

    def rotate(self, centre, theta=None):
        """
        Apply a rotation on this image with a centre
        :param centre: coordinate of the centre of the rotation in relation to the centre of this card
        :param theta: amount of rotation in radian (clockwise). If a range is given, rotate by a random amount within
        that range
        :return: none
        """
        if isinstance(theta, tuple) or (isinstance(theta, list) and len(theta) == 2):
            theta = random.uniform(theta[0], theta[1])

        # Rotation math
        self.x -= -centre[1] * math.sin(theta) + centre[0] * math.cos(theta)
        self.y -= centre[1] * math.cos(theta) + centre[0] * math.sin(theta)

        # Offset for the coordinate translation
        self.x += centre[0]
        self.y += centre[1]

        self.theta += theta
        pass


class ExtractedObject:
    """
    Simple struct to hold information about an extracted object
    """
    def __init__(self, label, key_pts):
        self.label = label
        self.key_pts = key_pts


def main():
    random.seed()

    img_bg = cv2.imread('data/frilly_0007.jpg')
    #img = cv2.imread('data/c16-143-burgeoning.png')

    generator = ImageGenerator(img_bg, [], 1440, 960)
    card_pool = pd.DataFrame()
    for set_name in fetch_data.all_set_list:
        df = fetch_data.load_all_cards_text('data/csv/%s.csv' % set_name)
        card_info = df.iloc[random.randint(0, df.shape[0] - 1)]
        # Currently ignoring planeswalker cards due to their different card layout
        is_planeswalker = 'Planeswalker' in card_info['type_line']
        if not is_planeswalker:
            card_pool = card_pool.append(card_info)

    for i in [random.randrange(0, card_pool.shape[0] - 1, 1) for _ in range(10)]:
        card_info = card_pool.iloc[i]
        img_name = '../usb/data/png/%s/%s_%s.png' % (card_info['set'], card_info['collector_number'],
                                                     fetch_data.get_valid_filename(card_info['name']))
        print(img_name)
        card_img = cv2.imread(img_name)
        if card_img is None:
            fetch_data.fetch_card_image(card_info, out_dir='../usb/data/png/%s' % card_info['set'])
            card_img = cv2.imread(img_name)
        card = Card(card_img, card_info, None)

        card.bind_to_generator(generator, x=random.uniform(0, generator.width), y=random.uniform(0, generator.height),
                               theta=0, scale=0.5)
        card.shift([-100, 100], [-100, 100])
        card.rotate((0, 0), [-math.pi, math.pi])
    generator.display()
    pass


if __name__ == '__main__':
    main()
