from train import *

model = object_detector.create(validation_data=val_data, model_spec=spec, batch_size=16, train_whole_model=True, epochs=100)

model.evaluate(val_data)

model.evaluate_tflite('android.tflite', val_data)