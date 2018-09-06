import os
from glob import glob
import random
from transform_data import data_dir, darknet_dir


def main():
    random.seed()
    data_list = []
    for subdir in glob('%s/train/*' % data_dir):
        for data in glob(subdir + "/*.jpg"):
            data_list.append(os.path.abspath(data))
    random.shuffle(data_list)

    test_ratio = 0.1
    test_list = data_list[:int(test_ratio * len(data_list))]
    train_list = data_list[int(test_ratio * len(data_list)):]
    with open('%s/train.txt' % darknet_dir, 'w') as train_txt:
        for data in train_list:
            train_txt.write(data + '\n')
    with open('%s/test.txt' % darknet_dir, 'w') as test_txt:
        for data in test_list:
            test_txt.write(data + '\n')
    return


if __name__ == '__main__':
    main()
