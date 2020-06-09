# EfficientDet JS

## A live webcam demo of (EfficientDet)[https://github.com/google/automl/tree/master/efficientdet] 


Commands to create the model. Requires that you get (automl)[https://github.com/google/automl/tree/master/efficientdet] running.

1. downlaod the checkpoint

    wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz

2. export the saved model

    python3 model_inspect.py --runmode=saved_model --model_name=efficientdet-d0 --ckpt_path=efficientdet-d0 

3. Export to tensorflow.js.

    tensorflowjs_converter --input_format=tf_saved_model --signature_name=serving_default --output_format=tfjs_graph_model  /tmp/saved_model/ websaved-d0 

4. Copy to dist

    cp -r websaved-d0 dist/d0 


### Misc.

This repository contains some tf.js keras layers, operations, and initializers that are not being used right now. This repo was originally a port of (this keras implementation)[https://github.com/xuannianz/EfficientDet] to JS before google released the automl repository with an official python implementation that can be exported as savedmodel.