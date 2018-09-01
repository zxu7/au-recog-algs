"""baseline system"""

import os
import keras
import numpy as np
import pandas as pd
from src.models import smallvgg
from src.keras_gen import ManifestGenerator
from functools import partial
from src.organize import WorkFlow, BaseConfig, parse_json

# config path
CONFIG_PATH = 'configs/emotionet1.config'


# base config
class Config(BaseConfig):
    def __init__(self):
        super(BaseConfig, self).__init__()
        self.config = {
            "train_manifest_path": None,  # str;
            "valid_manifest_path": '/hdd1/data/EmotioNet/annotations/valid_au_labels.csv',  # str;
            "aus": [1, 2, 4, 5, 6, 9, 12, 17, 20, 25, 26, 43],  # list; list of action units to predict
            "num_train_workers": 4,  # int; number of workers for processing data b4 sending to gpu
            "rescale": '1/255',  # str of math op; image array rescaling param
            "img_size": [128, 128],  # list; w, h
            "epochs": 5, # int;
        }


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
    # config and params
    config = Config()
    config.update(parse_json(CONFIG_PATH))
    TRAIN_MANIFEST_PATH = str(config.config['train_manifest_path'])
    VALID_MANIFEST_PATH = str(config.config['valid_manifest_path'])
    NUM_TRAIN_WORKERS = int(config.config['num_train_workers'])
    AUs = config.config['aus']
    RESCALE = eval(config.config['rescale'])
    IMG_SIZE = config.config['img_size']
    EPOCHS = int(config.config['epochs'])

    # workflow for saving training info
    workflow = WorkFlow()
    workflow.start(config=config)
    TRAIN_LOG_PATH = os.path.join(workflow.new_experiment_dir, 'training.log')
    MODEL_PATH = os.path.join(workflow.new_experiment_dir, 'model_15000.h5')

    # load data
    train_manifest = pd.read_csv(TRAIN_MANIFEST_PATH)
    train_manifest = train_manifest.head(15000)
    valid_manifest = pd.read_csv(VALID_MANIFEST_PATH)

    train_manifest_gen = ManifestGenerator(rescale=RESCALE, horizontal_flip=True)
    valid_manifest_gen = ManifestGenerator(rescale=RESCALE)
    train_gen = train_manifest_gen.flow_from_manifest(train_manifest,
                                                      partial(get_filenames, train=True),
                                                      get_labels,
                                                      target_size=IMG_SIZE)
    valid_gen = valid_manifest_gen.flow_from_manifest(valid_manifest,
                                                      partial(get_filenames, train=False),
                                                      get_labels,
                                                      target_size=IMG_SIZE,
                                                      shuffle=False)

    # train and save model
    # save loss, metrics history
    csvlogger = keras.callbacks.CSVLogger(TRAIN_LOG_PATH)
    callbacks = [csvlogger]
    model = smallvgg(IMG_SIZE[0], IMG_SIZE[1], classes=len(AUs))
    hist = model.fit_generator(train_gen,
                               epochs=EPOCHS,
                               validation_data=valid_gen,
                               workers=NUM_TRAIN_WORKERS,
                               callbacks=callbacks)
    model.save(MODEL_PATH)
