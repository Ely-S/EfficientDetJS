FROM tensorflow/tensorflow:2.3.0

# By default, download efficientdet-d0
# use --build-arg SIZE=d1 (d2 ,...d7) to build a larger, slower model
ARG SIZE=d0

WORKDIR /tmp

RUN apt-get update && apt-get install -y unzip wget git
RUN wget https://github.com/google/automl/archive/master.zip
RUN unzip master.zip

WORKDIR /tmp/automl-master/efficientdet
RUN pip3 install -r requirements.txt

RUN wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-${SIZE}.tar.gz
RUN tar -xzf efficientdet-${SIZE}.tar.gz

RUN python model_inspect.py --ckpt_path efficientdet-${SIZE} --runmode=freeze

RUN pip3 install tensorflowjs
RUN tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names detections \
    --output_format=tfjs_graph_model \
    --skip_op_check \
    /tmp/deff/savedmodel  /tmp/efficientdet-${SIZE}.js
