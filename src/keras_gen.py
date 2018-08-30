import keras
import os
import numpy as np
from keras import backend
from functools import partial
import multiprocessing
from keras.preprocessing.image import ImageDataGenerator, Iterator, load_img, img_to_array, array_to_img


class ManifestGenerator(ImageDataGenerator):
    def __init__(self, **kwargs):
        super(ManifestGenerator, self).__init__(**kwargs)

    def flow_from_manifest(self,
                           manifest, filename_func, label_func,
                           target_size=(256, 256), color_mode='rgb',
                           batch_size=32, shuffle=True, seed=None,
                           save_to_dir=None,
                           save_prefix='',
                           save_format='png',
                           follow_links=False,
                           subset=None,
                           interpolation='nearest'):
        return ManifestIterator(
            manifest, self, filename_func, label_func,
            target_size=target_size, color_mode=color_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)


class ManifestIterator(Iterator):
    def __init__(self, manifest, image_data_generator,
                 filename_func, label_func,
                 target_size=(256, 256), color_mode='rgb',
                 # classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest'):
        if data_format is None:
            data_format = backend.image_data_format()
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        # self.classes = classes
        # if class_mode not in {'categorical', 'binary', 'sparse',
        #                       'input', None}:
        #     raise ValueError('Invalid class_mode:', class_mode,
        #                      '; expected one of "categorical", '
        #                      '"binary", "sparse", "input"'
        #                      ' or None.')
        # self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError('Invalid subset name: ', subset,
                                 '; expected "training" or "validation"')
        else:
            split = None
        self.subset = subset

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp',
                              'ppm', 'tif', 'tiff'}
        # First, count the number of samples and classes.
        self.samples = 0
        self.manifest = manifest

        # if not classes:
        #     classes = []
        #     for subdir in sorted(os.listdir(directory)):
        #         if os.path.isdir(os.path.join(directory, subdir)):
        #             classes.append(subdir)
        # self.num_classes = len(classes)
        # self.class_indices = dict(zip(classes, range(len(classes))))

        # TODO: assert train_manifest files all exist
        # pool = multiprocessing.pool.ThreadPool()
        # function_partial = partial(_count_valid_files_in_directory,
        #                            white_list_formats=white_list_formats,
        #                            follow_links=follow_links,
        #                            split=split)
        # self.samples = sum(pool.map(function_partial,
        #                             (os.path.join(directory, subdir)
        #                              for subdir in classes)))
        filenames = filename_func(manifest)  # list of filename strings
        labels = label_func(manifest)  # a numpy.array of labels
        assert len(filenames) == len(labels), "filenames has length: {}, different from labels" \
                                              " lenghth: {}".format(len(filenames), len(labels))
        self.samples = len(filenames)
        print('Found %d images belonging to %d classes.' %
              (self.samples, labels.shape[-1]))

        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        self.filenames = filenames
        self.labels = labels
        # self.classes = np.zeros((self.samples,), dtype='int32')
        # i = 0
        # for dirpath in (os.path.join(directory, subdir) for subdir in classes):
        #     results.append(
        #         pool.apply_async(_list_valid_filenames_in_directory,
        #                          (dirpath, white_list_formats, split,
        #                           self.class_indices, follow_links)))
        # for res in results:
        #     classes, filenames = res.get()
        #     self.classes[i:i + len(classes)] = classes
        #     self.filenames += filenames
        #     i += len(classes)
        # pool.close()
        # pool.join()
        super(ManifestIterator, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=backend.floatx())
        batch_y = np.zeros(
            ((len(index_array),) + self.labels.shape[1:]),
            dtype=backend.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(fname,
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = self.labels[j]
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        # batch_y = np.zeros(((len(index_array),) + self.labels.shape[1:]))

        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

