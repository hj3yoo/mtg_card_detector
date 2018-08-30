import random
import math
import cv2
import numpy as np
import imutils
import pandas as pd
import fetch_data
import generate_data
from shapely import geometry

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

    def add_card(self, card, x=0, y=0, theta=0.0, scale=1.0):
        """
        Add a card to this generator scenario.
        :param card: card to be added
        :param x: new X-coordinate for the centre of the card
        :param y: new Y-coordinate for the centre of the card
        :param theta: new angle for the card
        :param scale: new scale for the card
        :return: none
        """
        self.cards.append(card)
        card.x = x
        card.y = y
        card.theta = theta
        card.scale = scale
        pass

    def display(self):
        """
        Display the current state of the generator
        :return: none
        """
        self.check_visibility()
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

            for ext_obj in card.objects:
                if ext_obj.visible:
                    for pt in ext_obj.key_pts:
                        cv2.circle(img_bg, card.coordinate_in_generator(pt[0], pt[1]), 2, (0, 0, 255), 2)
                    bounding_box = card.bb_in_generator(ext_obj.key_pts)
                    cv2.rectangle(img_bg, bounding_box[0], bounding_box[2], (0, 255, 0), 2)

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

    def check_visibility(self, visibility=0.5):
        """
        Check whether if extracted objects in each card are visible in the current scenario, and update their status
        :param visibility: minimum ratio of the object's area that aren't covered by another card to be visible
        :return: none
        """
        card_poly_list = [geometry.Polygon([card.coordinate_in_generator(0, 0),
                                            card.coordinate_in_generator(0, len(card.img)),
                                            card.coordinate_in_generator(len(card.img[0]), len(card.img)),
                                            card.coordinate_in_generator(len(card.img[0]), 0)]) for card in self.cards]

        # First card in the list is overlaid on the bottom of the card pile
        for i in range(len(self.cards)):
            card = self.cards[i]
            for ext_obj in card.objects:
                obj_poly = geometry.Polygon([card.coordinate_in_generator(pt[0], pt[1]) for pt in ext_obj.key_pts])
                obj_area = obj_poly.area
                # Check if the other cards are blocking this object
                for card_poly in card_poly_list[i + 1:]:
                    obj_poly = obj_poly.difference(card_poly)
                visible_area = obj_poly.area
                print("%s: %.1f visible" % (ext_obj.label, visible_area / obj_area))
                ext_obj.visible = obj_area * visibility <= visible_area

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
    def __init__(self, img, card_info, objects, x=None, y=None, theta=None, scale=None):
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
        self.x = x
        self.y = y
        self.theta = theta
        self.scale = scale
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

    def coordinate_in_generator(self, x, y):
        """
        Converting coordinate within the card into the coordinate in the generator it is associated with
        :param x: x coordinate within the card
        :param y: y coordinate within the card
        :return: (x, y) coordinate in the generator
        """
        # Relative distance in X & Y axis, if the centre of the card is at the origin (0, 0)
        rel_x = x - len(self.img[0]) // 2
        rel_y = y - len(self.img) // 2

        # Scaling
        rel_x *= self.scale
        rel_y *= self.scale

        # Rotation
        rot_x = rel_x - rel_y * math.sin(self.theta) + rel_x * math.cos(self.theta)
        rot_y = rel_y + rel_y * math.cos(self.theta) + rel_x * math.sin(self.theta)

        # Negate offset
        rot_x -= rel_x
        rot_y -= rel_y

        # Shift
        gen_x = rot_x + self.x
        gen_y = rot_y + self.y

        return int(gen_x), int(gen_y)

    def bb_in_generator(self, key_pts):
        """
        Convert a keypoints of bounding box in card into the coordinate in the generator
        :param key_pts: keypoints of the bounding box
        :return: bounding box represented by 4 points in the generator
        """
        x1 = -math.inf
        x2 = math.inf
        y1 = -math.inf
        y2 = math.inf
        for key_pt in key_pts:
            coord_in_gen = self.coordinate_in_generator(key_pt[0], key_pt[1])
            x1 = max(x1, coord_in_gen[0])
            x2 = min(x2, coord_in_gen[0])
            y1 = max(y1, coord_in_gen[1])
            y2 = min(y2, coord_in_gen[1])
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


class ExtractedObject:
    """
    Simple struct to hold information about an extracted object
    """
    def __init__(self, label, key_pts):
        self.label = label
        self.key_pts = key_pts
        self.visible = False


def main():
    random.seed()
    img_bg = cv2.imread('data/frilly_0007.jpg')
    generator = ImageGenerator(img_bg, [], 1440, 960)
    card_pool = pd.DataFrame()
    for set_name in fetch_data.all_set_list:
        df = fetch_data.load_all_cards_text('data/csv/%s.csv' % set_name)
        card_info = df.iloc[random.randint(0, df.shape[0] - 1)]
        # Currently ignoring planeswalker cards due to their different card layout
        is_planeswalker = 'Planeswalker' in card_info['type_line']
        if not is_planeswalker:
            card_pool = card_pool.append(card_info)
    a = 1
    for i in [random.randrange(0, card_pool.shape[0] - 1, 1) for _ in range(20)]:
        card_info = card_pool.iloc[i]
        img_name = '../usb/data/png/%s/%s_%s.png' % (card_info['set'], card_info['collector_number'],
                                                     fetch_data.get_valid_filename(card_info['name']))
        print(img_name)
        card_img = cv2.imread(img_name)
        if card_img is None:
            fetch_data.fetch_card_image(card_info, out_dir='../usb/data/png/%s' % card_info['set'])
            card_img = cv2.imread(img_name)
        detected_object_list = generate_data.apply_bounding_box(card_img, card_info)
        card = Card(card_img, card_info, detected_object_list)

        generator.add_card(card, x=random.uniform(200, generator.width - 200),
                           y=random.uniform(200, generator.height - 200), theta=random.uniform(-math.pi, math.pi), scale=0.5)
        #card.shift([-100, 100], [-100, 100])
        #card.rotate((0, 0), [-math.pi / 4, math.pi / 4])
        a += 1
    generator.display()
    pass


if __name__ == '__main__':
    main()
