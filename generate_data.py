from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import pickle
import math
import random
import os


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


def main():
    #bg_images = load_dtd()
    bg = Backgrounds()
    bg.get_random(display=True)
    return


if __name__ == '__main__':
    main()
