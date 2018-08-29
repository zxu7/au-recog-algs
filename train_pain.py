"""baseline system"""

import os
import numpy as np
import pandas as pd
from src.models import smallvgg
from src.keras_gen import ManifestGenerator

MANIFEST_PATH = '/hdd1/data/PAIN/processed/face_crops/data.csv'
AUs = [0, 4, 6, 7, 9, 10, 12, 15, 20, 25, 26, 27, 43, 50]


def get_filenames(manifest):
    """
    :param manifest: pd.DataFrame
    :return: list; list of filename strings
    """
    return manifest['img_path'].tolist()


def get_labels(manifest):
    """
    :param manifest: pd.DataFrame
    :return: numpy.array; labels represented in numpy.array
    """
    aus = np.array(manifest.iloc[:, -14:])
    out = np.array(np.array(aus) > 0, dtype='int')
    return out


if __name__ == '__main__':
    manifest = pd.read_csv(MANIFEST_PATH)

    train_manifest_gen = ManifestGenerator(rescale=1/255, horizontal_flip=True)
    train_gen = train_manifest_gen.flow_from_manifest(manifest, get_filenames, get_labels, target_size=(128,128))

    model = smallvgg(128, 128)
    model.fit_generator(train_gen, epochs=50)