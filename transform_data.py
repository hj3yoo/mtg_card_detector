

class ImageGenerator:
    """
    A template for generating a training image.
    """
    def __init__(self, img_bg, cards):
        """
        :param img_bg: background (textile) image
        :param cards: list of Card objects
        """
        self._img_bg = img_bg
        self._cards = cards
        self._img_result = None
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
    def __init__(self, img, card_info, objects, generator=None, x=None, y=None, theta=None):
        """
        :param img: image of the card
        :param card_info: details like name, mana cost, type, set, etc
        :param objects: list of ExtractedObjects like mana & set symbol, etc
        :param generator: ImageGenerator object that the card is bound to
        :param x: X-coordinate of the card's centre in relation to the generator
        :param y: Y-coordinate of the card's centre in relation to the generator
        :param theta: angle of rotation of the card in relation to the generator
        """
        self._img = img
        self._info = card_info
        self._objects = objects
        self._x = x
        self._y = y
        self._theta = theta
        pass

    def bind_to_generator(self, generator, x=0, y=0, theta=0):
        """
        Bind this card to an ImageGenerator object.
        :param generator: generator to be bound with
        :param x: new X-coordinate for the centre of the card
        :param y: new Y-coordinate for the centre of the card
        :param theta: new angle for the card
        :return: none
        """
        pass

    def shift(self, x=None, y=None):
        """
        Apply a X/Y translation on this image
        :param x: amount of X-translation. If range is given, translate by a random amount within that range
        :param y: amount of Y-translation. Refer to x when a range is given.
        :return: none
        """
        pass

    def rotate(self, centre, theta=None):
        """
        Apply a rotation on this image with a centre
        :param centre: coordinate of the centre of the rotation in relation to the centre of this card (self._x, self._y)
        :param theta: amount of rotation in radian. If a range is given, rotate by a random amount within that range
        :return: none
        """
        pass


class ExtractedObject:
    """
    Simple struct to hold information about an extracted object
    """
    def __init__(self, label, key_pts):
        self._label = label
        self._key_pts = key_pts


def main():
    pass


if __name__ == '__main__':
    main()