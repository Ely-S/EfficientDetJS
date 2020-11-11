###  Building the typescript

    yarn build

#### Building the model

There is a Dockerfile in hub that will an image containing an EfficientDet model and convert it to the tensorflow.js format

    cd hub
    docker build -t efficientdet-model-d0 .
    
    # Or to build a different moodel size
    # docker build --build-arg SIZE=d1 -t efficientdet-model-d0 .

    # Copy exported model files into current directory
    docker run -v (pwd):/out efficientdet-model-d0  cp -r /tmp/efficientdet-d0.js /out/
