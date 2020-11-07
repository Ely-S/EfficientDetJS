FROM tensorflow/tensorflow:2.3.0

WORKDIR /tmp

RUN apt-get update && apt-get install -y unzip wget git
RUN wget https://github.com/google/automl/archive/master.zip
RUN unzip master.zip

WORKDIR /tmp/automl-master/efficientdet
RUN pip3 install -r requirements.txt

RUN wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz
RUN tar -xzf efficientdet-d0.tar.gz

RUN python model_inspect.py --ckpt_path efficientdet-d0 --runmode=freeze

RUN pip3 install tensorflowjs
RUN tensorflowjs_converter  --input_format=tf_saved_model  --output_node_names detections --output_format=tfjs_graph_model  /tmp/deff/savedmodel  websaved --skip_op_check

