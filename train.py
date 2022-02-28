import numpy as np
import os

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf
assert tf.__version__.startswith('2') # IF you have problems comment this line

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

#SET THE CLASSES CREATED WITH THE LABELS IN TRAIN AND VALIDATE. EXAMPLE:['ape','mucca' ecc.]
train_data = object_detector.DataLoader.from_pascal_voc(
    'Images/train',
    'Images/train',
    ['', '']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'Images/validate',
    'Images/validate',
    ['', '']
)
#

#YOU CAN USE VARIOUS PRE-TRAINED MODELS:
#efficientdet-lite0 --> Average Precision 25.69%
#efficientdet-lite1 --> Average Precision 30.55%
#efficientdet-lite2 --> Average Precision 33.97%
#efficientdet-lite3 --> Average Precision 37.70%
#efficientdet-lite4 --> Average Precision 41.96% (very slow)
spec = model_spec.get('efficientdet_lite0')

#YOU CAN SET THE WISHED VALUES FOR A GOOD TRAINING
model = object_detector.create(train_data, model_spec=spec, batch_size=16, train_whole_model=True, epochs=500)

#model.evaluate(val_data)

model.export(export_dir='.', tflite_filename='android.tflite')

#model.evaluate_tflite('android.tflite', val_data)
