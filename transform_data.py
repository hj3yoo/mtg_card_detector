import argparse
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import imutils
import math
import numpy as np
import os
import pandas as pd
import random
from shapely import geometry

import fetch_data
import generate_data
from config import Config


def key_pts_to_yolo(key_pts, w_img, h_img):
    """
    Convert a list of keypoints into a yolo training format
    :param key_pts: list of keypoints
    :param w_img: width of the entire image
    :param h_img: height of the entire image
    :return: <x> <y> <width> <height>
    """
    x1 = max(0, min([pt[0] for pt in key_pts]))
    x2 = min(w_img, max([pt[0] for pt in key_pts]))
    y1 = max(0, min([pt[1] for pt in key_pts]))
    y2 = min(h_img, max([pt[1] for pt in key_pts]))
    x = (x2 + x1) / 2 / w_img
    y = (y2 + y1) / 2 / h_img
    width = (x2 - x1) / w_img
    height = (y2 - y1) / h_img
    return x, y, width, height


class ImageGenerator:
    """
    A template for generating a training image
    An ImageGenerator contains a background image, list of cards, and other environmental parameters to
    set up a training image for YOLO network
    """
    def __init__(self, img_bg, class_ids, width, height, skew=None, cards=None):
        """
        :param img_bg: background (textile) image
        :param width: width of the training image
        :param height: height of the training image
        :param skew: 4 coordinates that indicates the corners (in normalized form) for perspective transform
        :param cards: list of Card objects
        """
        self.img_bg = img_bg
        self.class_ids = class_ids
        self.img_result = None
        self.width = width
        self.height = height
        if cards is None:
            self.cards = []
        else:
            self.cards = cards

        # Compute transform matrix for perspective transform (used for skewing the final result)
        if skew is not None:
            orig_corner = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)
            new_corner = np.array([[width * s[0], height * s[1]] for s in skew], dtype=np.float32)
            self.M = cv2.getPerspectiveTransform(orig_corner, new_corner)
            pass
        else:
            self.M = None
        pass

    def add_card(self, card, x=None, y=None, theta=0.0, scale=1.0):
        """
        Add a card to this generator scenario.
        :param card: card to be added
        :param x: new X-coordinate for the centre of the card
        :param y: new Y-coordinate for the centre of the card
        :param theta: new angle for the card
        :param scale: new scale for the card
        :return: none
        """
        # If the position isn't given, push it out of the image so that it won't be visible during rendering
        if x is None:
            x = -len(card.img[0]) / 2
        if y is None:
            y = -len(card.img) / 2
        self.cards.append(card)
        card.x = x
        card.y = y
        card.theta = theta
        card.scale = scale
        pass

    def render(self, visibility=0.5, aug=None, display=False, debug=False):
        """
        Display the current state of the generator.
        :param visibility: portion of the card's image that must not be overlapped by other cards for the card to be
                           considered as visible
        :param aug: image augmentator to apply during rendering
        :param display: flag for displaying the rendering result
        :param debug: flag for debug
        :return: none
        """
        self.check_visibility(visibility=visibility)
        img_result = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        card_mask = cv2.imread(Config.card_mask_path)

        for card in self.cards:
            card_x = int(card.x + 0.5)
            card_y = int(card.y + 0.5)

            # Scale & rotate card image
            img_card = cv2.resize(card.img, (int(len(card.img[0]) * card.scale), int(len(card.img) * card.scale)))
            # Add a random glaring on individual card - it happens frequently in real life as MTG cards can reflect
            # the lights very well.
            if aug is not None:
                seq = iaa.Sequential([
                    iaa.SimplexNoiseAlpha(first=iaa.Add(random.randrange(128)), size_px_max=[1, 3],
                                          upscale_method="cubic"),  # Lighting
                ])
                img_card = seq.augment_image(img_card)
            mask_scale = cv2.resize(card_mask, (int(len(card_mask[0]) * card.scale), int(len(card_mask) * card.scale)))
            img_mask = cv2.bitwise_and(img_card, mask_scale)
            img_rotate = imutils.rotate_bound(img_mask, card.theta / math.pi * 180)
            
            # Calculate the position of the card image in relation to the background
            # Crop the card image if it's out of boundary
            card_w = len(img_rotate[0])
            card_h = len(img_rotate)
            card_crop_x1 = max(0, card_w // 2 - card_x)
            card_crop_x2 = min(card_w, card_w // 2 + len(img_result[0]) - card_x)
            card_crop_y1 = max(0, card_h // 2 - card_y)
            card_crop_y2 = min(card_h, card_h // 2 + len(img_result) - card_y)
            img_card_crop = img_rotate[card_crop_y1:card_crop_y2, card_crop_x1:card_crop_x2]

            # Calculate the position of the corresponding area in the background
            bg_crop_x1 = max(0, card_x - (card_w // 2))
            bg_crop_x2 = min(len(img_result[0]), int(card_x + (card_w / 2) + 0.5))
            bg_crop_y1 = max(0, card_y - (card_h // 2))
            bg_crop_y2 = min(len(img_result), int(card_y + (card_h / 2) + 0.5))
            img_result_crop = img_result[bg_crop_y1:bg_crop_y2, bg_crop_x1:bg_crop_x2]

            # Override the background with the current card
            img_result_crop = np.where(img_card_crop, img_card_crop, img_result_crop)
            img_result[bg_crop_y1:bg_crop_y2, bg_crop_x1:bg_crop_x2] = img_result_crop
            
            if debug:
                for ext_obj in card.objects:
                    if ext_obj.visible:
                        for pt in ext_obj.key_pts:
                            cv2.circle(img_result, card.coordinate_in_generator(pt[0], pt[1]), 2, (1, 1, 255), 10)
                        bounding_box = card.bb_in_generator(ext_obj.key_pts)
                        cv2.rectangle(img_result, bounding_box[0], bounding_box[2], (1, 255, 1), 5)

        img_result = cv2.GaussianBlur(img_result, (5, 5), 0)

        # Skew the cards if it's provided
        if self.M is not None:
            img_result = cv2.warpPerspective(img_result, self.M, (self.width, self.height))
            if debug:
                for card in self.cards:
                    for ext_obj in card.objects:
                        if ext_obj.visible:
                            new_pts = np.array([[list(card.coordinate_in_generator(pt[0], pt[1]))]
                                                for pt in ext_obj.key_pts], dtype=np.float32)
                            new_pts = cv2.perspectiveTransform(new_pts, self.M)
                            for pt in new_pts:
                                cv2.circle(img_result, (pt[0][0], pt[0][1]), 2, (255, 1, 1), 10)

        img_bg = cv2.resize(self.img_bg, (self.width, self.height))
        img_result = np.where(img_result, img_result, img_bg)

        # Apply image augmentation
        if aug is not None:
            img_result = aug.augment_image(img_result)

        if display or debug:
            cv2.imshow('Result', img_result)
            cv2.waitKey(0)

        self.img_result = img_result
        pass

    def generate_horizontal_span(self, gap=None, scale=None, theta=0, shift=None, jitter=None):
        """
        Generating the first scenario where the cards are laid out in a straight horizontal line
        :param gap: horizontal offset between each adjacent cards
        :param scale: scale of each cards in the generator
        :param theta: rotation of the entire span in radian
        :param shift: range of arbitrary offset for each card
        :param jitter: range of in-place rotation for each card in radian
        :return: True if successfully generated, otherwise False
        """
        # Set scale of the cards, variance of shift & jitter to be applied if they're not given
        card_size = (len(self.cards[0].img[0]), len(self.cards[0].img))
        if scale is None:
            # Scale the cards so that card takes about 50% of the image's height
            coverage_ratio = 0.5
            scale = self.height * coverage_ratio / card_size[1]
        if shift is None:
            # Plus minus 5% of the card's height
            shift = [-card_size[1] * scale * 0.05, card_size[1] * scale * 0.05]
            pass
        if jitter is None:
            # Plus minus 10 degrees
            jitter = [-math.pi / 18, math.pi / 18]
        if gap is None:
            # 25% of the card's width - set symbol and 1-2 mana symbols will be visible on each card
            gap = card_size[0] * scale * 0.4

        # Determine the location of the first card
        # The cards will cover (width of a card + (# of cards - 1) * gap) pixels wide and (height of a card) pixels high
        x_anchor = int(self.width / 2 + (len(self.cards) - 1) * gap / 2)
        y_anchor = self.height // 2
        for card in self.cards:
            card.scale = scale
            card.x = x_anchor
            card.y = y_anchor
            card.theta = 0
            card.shift(shift, shift)
            card.rotate(jitter)
            card.rotate(theta, centre=(self.width // 2 - x_anchor, self.height // 2 - y_anchor))
            x_anchor -= gap

        return True

    def generate_vertical_span(self, gap=None, scale=None, theta=0, shift=None, jitter=None):
        """
        Generating the second scenario where the cards are laid out in a straight vertical line
        :param gap: horizontal offset between each adjacent cards
        :param scale: scale of each cards in the generator
        :param theta: rotation of the entire span in radian
        :param shift: range of arbitrary offset for each card
        :param jitter: range of in-place rotation for each card in radian
        :return: True if successfully generated, otherwise False
        :return: True if successfully generated, otherwise False
        """
        # Set scale of the cards, variance of shift & jitter to be applied if they're not given
        card_size = (len(self.cards[0].img[0]), len(self.cards[0].img))
        if scale is None:
            # Scale the cards so that card takes about 50% of the image's height
            coverage_ratio = 0.5
            scale = self.height * coverage_ratio / card_size[1]
        if shift is None:
            # Plus minus 5% of the card's height
            shift = [-card_size[1] * scale * 0.05, card_size[1] * scale * 0.05]
            pass
        if jitter is None:
            # Plus minus 5 degrees
            jitter = [-math.pi / 36, math.pi / 36]
        if gap is None:
            # 15% of the card's height - the title bar (with mana symbols) will be visible
            gap = card_size[1] * scale * 0.25

        # Determine the location of the first card
        # The cards will cover (width of a card) pixels wide and (height of a card + (# of cards - 1) * gap) pixels high
        x_anchor = self.width // 2
        y_anchor = int(self.height / 2 - (len(self.cards) - 1) * gap / 2)
        for card in self.cards:
            card.scale = scale
            card.x = x_anchor
            card.y = y_anchor
            card.theta = 0
            card.shift(shift, shift)
            card.rotate(jitter)
            card.rotate(theta, centre=(self.width // 2 - x_anchor, self.height // 2 - y_anchor))
            y_anchor += gap
        return True

    def generate_fan_out(self, centre, theta_between_cards=None, scale=None, shift=None, jitter=None):
        """
        Generating the third scenario where the cards are laid out in a fan shape
        :return: True if successfully generated, otherwise False
        """
        # TODO
        return False

    def generate_non_obstructive(self, tolerance=0.90, scale=None):
        """
        Generating the fourth scenario where the cards are laid in arbitrary position that doesn't obstruct other cards
        :param tolerance: minimum level of visibility for each cards
        :param scale: scale of each cards in generator
        :return: True if successfully generated, otherwise False
        """
        card_size = (len(self.cards[0].img[0]), len(self.cards[0].img))
        if scale is None:
            # Total area of the cards should cover about 25-40% of the entire image, depending on the number of cards
            scale = math.sqrt(self.width * self.height * min(0.25 + 0.02 * len(self.cards), 0.4)
                              / (card_size[0] * card_size[1] * len(self.cards)))
        # Position each card at random location that doesn't obstruct other cards
        i = 0
        while i < len(self.cards):
            card = self.cards[i]
            card.scale = scale
            rep = 0
            while True:
                card.x = random.uniform(card_size[1] * scale / 2, self.width - card_size[1] * scale)
                card.y = random.uniform(card_size[1] * scale / 2, self.height - card_size[1] * scale)
                card.theta = random.uniform(-math.pi, math.pi)
                self.check_visibility(self.cards[:i + 1], visibility=tolerance)
                # This position is not obstructive if all of the cards are visible
                is_visible = [other_card.objects[0].visible for other_card in self.cards[:i + 1]]
                non_obstructive = all(is_visible)
                if non_obstructive:
                    i += 1
                    break
                rep += 1
                if rep >= 1000:
                    # Reassign previous card's position
                    i -= 1
                    break
        return True

    def check_visibility(self, cards=None, i_check=None, visibility=0.5):
        """
        Check whether if extracted objects in a card is visible in the current scenario, and update their status
        :param cards: list of cards (in a correct Z-order). All cards in this Generator are checked by default.
        :param i_check: indices of cards that needs to be checked. Cards that aren't in this list will only be used
        to check visibility of other cards. All cards are checked by default.
        :param visibility: minimum ratio of the object's area that aren't covered by another card to be visible
        :return: none
        """
        if cards is None:
            cards = self.cards
        if i_check is None:
            i_check = range(len(cards))

        # Create a polygon of each card
        card_poly_list = [geometry.Polygon([card.coordinate_in_generator(0, 0),
                                            card.coordinate_in_generator(0, len(card.img)),
                                            card.coordinate_in_generator(len(card.img[0]), len(card.img)),
                                            card.coordinate_in_generator(len(card.img[0]), 0)]) for card in self.cards]
        template_poly = geometry.Polygon([(0, 0), (self.width, 0), (self.width, self.height), (0, self.height)])

        # First card in the list is overlaid on the bottom of the card pile
        for i in i_check:
            card = cards[i]
            for ext_obj in card.objects:
                obj_poly = geometry.Polygon([card.coordinate_in_generator(pt[0], pt[1]) for pt in ext_obj.key_pts])
                obj_area = obj_poly.area
                # Check if the other cards are blocking this object or if it's out of the template
                # If there are other polygons with higher indices in the list, that card is overlapping this object
                # We assume that no objects from the same card is on top of each other
                for card_poly in card_poly_list[i + 1:]:
                    obj_poly = obj_poly.difference(card_poly)
                obj_poly = obj_poly.intersection(template_poly)
                visible_area = obj_poly.area
                ext_obj.visible = obj_area * visibility <= visible_area

    def export_training_data(self, out_name, visibility=0.5, aug=None):
        """
        Export the generated training image along with the txt file for all bounding boxes
        :param out_name: path of the output file (without extension)
        :param visibility: portion of the card's image that must not be overlapped by other cards for the card to be
                           considered as visible
        :param aug: image augmentator to be applied
        :return: none
        """
        self.render(visibility, aug=aug)
        cv2.imwrite(out_name + '.jpg', self.img_result)
        out_txt = open(out_name + '.txt', 'w')
        for card in self.cards:
            for ext_obj in card.objects:
                if not ext_obj.visible:
                    continue
                coords_in_gen = [card.coordinate_in_generator(key_pt[0], key_pt[1]) for key_pt in ext_obj.key_pts]
                obj_yolo_info = key_pts_to_yolo(coords_in_gen, self.width, self.height)
                if ext_obj.label == 'card':
                    #class_id = self.class_ids[card.info['name']]
                    class_id = 0  # since only the entire card is used
                    out_txt.write(str(class_id) + ' %.6f %.6f %.6f %.6f\n' % obj_yolo_info)
        out_txt.close()


class Card:
    """
    A class for storing required information about a card in relation to the ImageGenerator
    """
    def __init__(self, img, card_info, objects, x=None, y=None, theta=None, scale=None):
        """
        :param img: image of the card
        :param card_info: details like name, mana cost, type, set, etc
        :param objects: list of ExtractedObjects like mana & set symbol, etc
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
        :param y: amount of Y-translation. If range is given, translate by a random amount within that range
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

    def rotate(self, theta, centre=(0, 0)):
        """
        Apply a rotation on this image with a centre
        :param theta: amount of rotation in radian (clockwise). If a range is given, rotate by a random amount within
                      that range
        :param centre: coordinate of the centre of the rotation in relation to the centre of this card
        :return: none
        """
        if isinstance(theta, tuple) or (isinstance(theta, list) and len(theta) == 2):
            theta = random.uniform(theta[0], theta[1])

        # If the centre given is the centre of this card, the whole math simplifies a bit
        # (This still works without the if statement, but let's not do useless trigs if we know the answer already)
        if centre is not (0, 0):
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
        coords_in_gen = [self.coordinate_in_generator(key_pt[0], key_pt[1]) for key_pt in key_pts]
        x1 = min([pt[0] for pt in coords_in_gen])
        x2 = max([pt[0] for pt in coords_in_gen])
        y1 = min([pt[1] for pt in coords_in_gen])
        y2 = max([pt[1] for pt in coords_in_gen])
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


class ExtractedObject:
    """
    Simple struct to hold information about an extracted object
    """
    def __init__(self, label, key_pts):
        self.label = label
        self.key_pts = key_pts
        self.visible = False


def main(args):
    random.seed()
    ia.seed(random.randrange(10000))

    bg_images = generate_data.load_dtd(dtd_dir='%s/dtd/images' % Config.data_dir, dump_it=False)
    background = generate_data.Backgrounds(images=bg_images)

    card_pool = pd.DataFrame()
    for set_name in Config.all_set_list:
        df = fetch_data.load_all_cards_text('%s/csv/%s.csv' % (Config.data_dir, set_name))
        card_pool = card_pool.append(df)
    class_ids = {}
    with open('%s/obj.names' % Config.data_dir) as names_file:
        class_name_list = names_file.read().splitlines()
        for i in range(len(class_name_list)):
            class_ids[class_name_list[i]] = i

    for i in range(args.num_gen):
        # Arbitrarily select top left and right corners for perspective transformation
        # Since the training image are generated with random rotation, don't need to skew all four sides
        skew = [[random.uniform(0, 0.25), 0], [0, 1], [1, 1],
                [random.uniform(0.75, 1), 0]]
        generator = ImageGenerator(background.get_random(), class_ids, args.width, args.height, skew=skew)
        out_name = ''

        # Use 2 to 5 cards per generator
        for _, card_info in card_pool.sample(random.randint(2, 5)).iterrows():
            img_name = '%s/card_img/png/%s/%s_%s.png' % (Config.data_dir, card_info['set'],
                                                         card_info['collector_number'],
                                                         fetch_data.get_valid_filename(card_info['name']))
            out_name += '%s%s_' % (card_info['set'], card_info['collector_number'])
            card_img = cv2.imread(img_name)
            if card_img is None:
                fetch_data.fetch_card_image(card_info, out_dir='%s/card_img/png/%s' % (Config.data_dir,
                                                                                       card_info['set']))
                card_img = cv2.imread(img_name)
            if card_img is None:
                print('WARNING: card %s is not found!' % img_name)
            detected_object_list = generate_data.apply_bounding_box(card_img, card_info)
            card = Card(card_img, card_info, detected_object_list)
            generator.add_card(card)

        for j in range(args.num_iter):
            seq = iaa.Sequential([
                iaa.Multiply((0.8, 1.2)),  # darken / brighten the whole image
                iaa.SimplexNoiseAlpha(first=iaa.Add(random.randrange(64)), per_channel=0.1, size_px_max=[3, 6],
                                      upscale_method="cubic"),  # Lighting
                iaa.AdditiveGaussianNoise(scale=random.uniform(0, 0.05) * 255, per_channel=0.1),  # Noises
                iaa.Dropout(p=[0, 0.05], per_channel=0.1)  # Dropout
            ])

            if i % 3 == 0:
                generator.generate_non_obstructive()
                generator.export_training_data(visibility=0.0, out_name='%s/train/non_obstructive_update/%s%d'
                                                                        % (Config.data_dir, out_name, j), aug=seq)
            elif i % 3 == 1:
                generator.generate_horizontal_span(theta=random.uniform(-math.pi, math.pi))
                generator.export_training_data(visibility=0.0, out_name='%s/train/horizontal_span_update/%s%d'
                                                                        % (Config.data_dir, out_name, j), aug=seq)
            else:
                generator.generate_vertical_span(theta=random.uniform(-math.pi, math.pi))
                generator.export_training_data(visibility=0.0, out_name='%s/train/vertical_span_update/%s%d'
                                                                        % (Config.data_dir, out_name, j), aug=seq)

            #generator.generate_horizontal_span(theta=random.uniform(-math.pi, math.pi))
            #generator.render(display=True, aug=seq, debug=True)
            print('Generated %s%d' % (out_name, j))
            generator.img_bg = background.get_random()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_gen', dest='num_gen', help='Number of training images to generate',
                        type=int, required=True)
    parser.add_argument('-ni', '--num_iter', dest='num_iter', help='Number of iterations to generate each config',
                        type=int, default=1)
    parser.add_argument('-w', '--width', dest='width', help='Width of the training image', type=int, default=1440)
    parser.add_argument('-ht', '--height', dest='height', help='Height of the training image', type=int, default=960)
    args = parser.parse_args()
    main(args)
