import random
import math
import cv2
import numpy as np
import imutils
import pandas as pd
import fetch_data
import generate_data
from shapely import geometry
import pytesseract

card_mask = cv2.imread('data/mask.png')


def key_pts_to_yolo(key_pts, w_img, h_img):
    """
    Convert a list of keypoints into a yolo training format
    :param key_pts: list of keypoints
    :param w_img: width of the entire image
    :param h_img: height of the entire image
    :return: <x> <y> <width> <height>
    """
    x1 = min([pt[0] for pt in key_pts])
    x2 = max([pt[0] for pt in key_pts])
    y1 = min([pt[1] for pt in key_pts])
    y2 = max([pt[1] for pt in key_pts])
    x = (x2 + x1) / 2 / w_img
    y = (y2 + y1) / 2 / h_img
    width = (x2 - x1) / w_img
    height = (y2 - y1) / h_img
    return x, y, width, height


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

    def display(self, debug=False):
        """
        Display the current state of the generator
        :return: none
        """
        self.check_visibility()
        img_result = cv2.resize(self.img_bg, (self.width, self.height))

        for card in self.cards:
            if card.x == 0.0 and card.y == 0.0 and card.theta == 0.0 and card.scale == 1.0:
                continue
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
                            cv2.circle(img_result, card.coordinate_in_generator(pt[0], pt[1]), 2, (0, 0, 255), 2)
                        bounding_box = card.bb_in_generator(ext_obj.key_pts)
                        cv2.rectangle(img_result, bounding_box[0], bounding_box[2], (0, 255, 0), 2)

        try:
            text = pytesseract.image_to_string(img_result, output_type=pytesseract.Output.DICT)
            print(text)
        except pytesseract.pytesseract.TesseractError:
            pass
        img_result = cv2.GaussianBlur(img_result, (5, 5), 0)
        cv2.imshow('Result', img_result)
        cv2.waitKey(0)
        self.img_result = img_result
        pass

    def generate_horizontal_span(self, gap=None, scale=None, shift=None, jitter=None):
        """
        Generating the first scenario where the cards are laid out in a straight horizontal line
        :return: none
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
            jitter = [-math.pi / 18, math.pi / 18]  # Plus minus 10 degrees
        if gap is None:
            # 25% of the card's width - set symbol and 1-2 mana symbols will be visible on each card
            gap = card_size[0] * scale * 0.25

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
            x_anchor -= gap
        pass

    def generate_vertical_span(self, gap=None, scale=None, shift=None, jitter=None):
        """
        Generating the second scenario where the cards are laid out in a straight vertical line
        :return: none
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
            gap = card_size[1] * scale * 0.15

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
            y_anchor += gap
        pass

        pass

    def generate_fan_out(self, centre, theta_between_cards=None, scale=None, shift=None, jitter=None):
        """
        Generating the third scenario where the cards are laid out in a fan shape
        :return: none
        """
        pass

    def generate_non_obstructive(self, tolerance=0.85, scale=None):
        """
        Generating the fourth scenario where the cards are laid in arbitrary position that doesn't obstruct other cards
        :param tolerance: minimum level of visibility for each cards
        :return:
        """
        card_size = (len(self.cards[0].img[0]), len(self.cards[0].img))
        if scale is None:
            # Total area of the cards should cover about 25-40% of the entire image, depending on the number of cards
            scale = math.sqrt(self.width * self.height * min(0.25 + 0.02 * len(self.cards), 0.4)
                              / (card_size[0] * card_size[1] * len(self.cards)))
        # Position each card at random location that doesn't obstruct other cards
        for i in range(len(self.cards)):
            card = self.cards[i]
            card.scale = scale
            while True:
                card.x = random.uniform(card_size[1] * scale / 2, self.width - card_size[1] * scale)
                card.y = random.uniform(card_size[1] * scale / 2, self.height - card_size[1] * scale)
                card.theta = random.uniform(-math.pi, math.pi)
                self.check_visibility(self.cards[:i + 1], visibility=tolerance)
                # This position is not obstructive if all of the cards are visible
                is_visible = [other_card.objects[0].visible for other_card in self.cards[:i + 1]]
                non_obstructive = all(is_visible)
                if non_obstructive:
                    break

    def check_visibility(self, cards=None, i_check=None, visibility=0.5):
        """
        Check whether if extracted objects in each card are visible in the current scenario, and update their status
        :param cards: list of cards (in a correct order)
        :param i_check: indices of cards that needs to be checked. Cards that aren't in this list will only be used
        to check visibility of other cards. All cards are checked by default.
        :param visibility: minimum ratio of the object's area that aren't covered by another card to be visible
        :return: none
        """
        if cards is None:
            cards = self.cards
        if i_check is None:
            i_check = range(len(cards))
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
                for card_poly in card_poly_list[i + 1:]:
                    obj_poly = obj_poly.difference(card_poly)
                obj_poly = obj_poly.intersection(template_poly)
                visible_area = obj_poly.area
                #print(visible_area, obj_area, len(card.img[0]) * len(card.img) * card.scale * card.scale)
                #print("%s: %.1f visible" % (ext_obj.label, visible_area / obj_area * 100))
                ext_obj.visible = obj_area * visibility <= visible_area

    def export_training_data(self, out_name):
        """
        Export the generated training image along with the txt file for all bounding boxes
        :return: none
        """
        cv2.imwrite(out_name + '.jpg', self.img_result)
        out_txt = open(out_name+ '.txt', 'w')
        for card in self.cards:
            for ext_obj in card.objects:
                if not ext_obj.visible:
                    continue
                coords_in_gen = [card.coordinate_in_generator(key_pt[0], key_pt[1]) for key_pt in ext_obj.key_pts]
                obj_yolo_info = key_pts_to_yolo(coords_in_gen, self.width, self.height)
                if ext_obj.label == 'card':
                    out_txt.write('0 %.6f %.6f %.6f %.6f\n' % obj_yolo_info)
                    pass
                elif ext_obj.label[:ext_obj.label.find[':']] == 'mana_symbol':
                    # TODO
                    pass
                elif ext_obj.label[:ext_obj.label.find[':']] == 'set_symbol':
                    # TODO
                    pass
        out_txt.close()
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

    def rotate(self, theta, centre=(0, 0)):
        """
        Apply a rotation on this image with a centre
        :param theta: amount of rotation in radian (clockwise). If a range is given, rotate by a random amount within
        :param centre: coordinate of the centre of the rotation in relation to the centre of this card
        that range
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
        '''
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
        '''
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
    for i in [random.randrange(0, card_pool.shape[0] - 1, 1) for _ in range(4)]:
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

        generator.add_card(card)
        #generator.add_card(card, x=random.uniform(200, generator.width - 200),
        #                   y=random.uniform(200, generator.height - 200), theta=random.uniform(-math.pi, math.pi), scale=0.5)
        #card.shift([-100, 100], [-100, 100])
        #card.rotate((0, 0), [-math.pi / 4, math.pi / 4])
    import time

    for i in range(100):
        generator.generate_vertical_span()
        generator.display(debug=False)
        generator.export_training_data(out_name='data/test')
    #generator.generate_horizontal_span()
    #generator.display(debug=True)
    #generator.generate_vertical_span()
    #generator.display(debug=True)
    pass


if __name__ == '__main__':
    main()
