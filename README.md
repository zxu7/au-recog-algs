# Action Unit Recognition (Under Development)

### Requirements
keras >= 2.2.2

### Data pipeline
This repo uses `ManifestGenerator` to load/process data. Example:

```
from src.keras_gen import ManifestGenerator
 
# initialize ManifestGenerator with preprocessing, augmentation parameters 
train_manifest_gen = ManifestGenerator(rescale=1/255, horizontal_flip=True)
 
# manifest can be a pandas DataFrame object or anything similar, you name it!
# get_filenames is a callable function that returns list of filenames from manifest
# get_labels is a callable function that returns list of labels from manifest
# train_gen can be used in model.fit_generator()
train_gen = train_manifest_gen.flow_from_manifest(manifest, get_filenames, get_labels, target_size=(128,128))

```

The `train_pain.py` is a working training script can successfully overfit the PAIN dataset. Check it out!

### Models