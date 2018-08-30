"""baseline system"""

import os
import numpy as np
import pandas as pd
from src.models import smallvgg
from src.keras_gen import ManifestGenerator
from functools import partial

TRAIN_MANIFEST_PATH = '/hdd1/data/EmotioNet/annotations/train_au_labels.csv'
VALID_MANIFEST_PATH = '/hdd1/data/EmotioNet/annotations/valid_au_labels.csv'
AUs = [1, 2, 4, 5, 6, 9, 12, 17, 20, 25, 26, 43]


def get_filenames(manifest, train=True):
    """
    :param manifest: pd.DataFrame
    :param train: bool;
    :return: list; list of filename strings
    """
    urls = manifest['url'].tolist()
    prefix = '/hdd1/data/EmotioNet/train_images/' if train else '/hdd1/data/EmotioNet/valid_images/'
    filenames = [prefix + os.path.basename(url) for url in urls]
    return filenames


def get_labels(manifest):
    """
    convert 0, 999 to 0; 1 to 1
    :param manifest: pd.DataFrame
    :return: numpy.array; labels represented in numpy.array
    """
    aus = np.array(manifest.iloc[:, -len(AUs):])
    out = np.array(np.array(aus) == 1, dtype='int')
    return out


if __name__ == '__main__':
    train_manifest = pd.read_csv(TRAIN_MANIFEST_PATH)
    train_manifest = train_manifest.head(15000)
    valid_manifest = pd.read_csv(VALID_MANIFEST_PATH)

    train_manifest_gen = ManifestGenerator(rescale=1 / 255, horizontal_flip=True)
    valid_manifest_gen = ManifestGenerator(rescale=1 / 255)
    train_gen = train_manifest_gen.flow_from_manifest(train_manifest, partial(get_filenames, train=True),
                                                      get_labels, target_size=(128, 128))
    valid_gen = valid_manifest_gen.flow_from_manifest(valid_manifest, partial(get_filenames, train=False),
                                                      get_labels, target_size=(128, 128),
                                                      shuffle=False)

    model = smallvgg(128, 128, classes=len(AUs))
    model.fit_generator(train_gen, epochs=20, validation_data=valid_gen)
    # model.save('trial1.h5')
