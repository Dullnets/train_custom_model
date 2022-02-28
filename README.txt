This python script allows you to drag a custom model with a pre-trained model.

I used:
-python 3.9.10
-tensorflow 2.8.0
-tflite-model-maker 0.3.4
-tflite-support 0.3.1

For the images labels I used labelimg which generated an .xml file to put together with the images in the respective folders.
You can do the evaluate in the train class but the process becomes slower.
Example:
model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)