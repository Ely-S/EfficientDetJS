### EfficientDetJs

This library lets you easily deploy a State-of-the-Art EfficientDet model to a tensorflow.js runtime. EfficientDet is a fast, very powerful neural architecture with an active open source [implementation](https://github.com/google/automl) making it a good base for new projects. This library allows you to use a pretrained or custom EfficientDet model without messing with tensorflow.js yourself.

[See a live demo](https://ondaka.github.io/EfficientDetJS/example/dist/).

## Getting Started


        const model = new EfficientDet()

        // load the model from tf hub
        await model.load()

        // get an array of object bounding boxes
        // .predict accepts a a Tensor3D of an image.
        const predictions = model.predict(image)

        // draw boxes on canvas
        model.draw(predictions, document.getElementById("mycanvas))

### Pretrained checkpoint

The pretraiend checkpoint, efficientdet-d0 is trained on a 90 class COCO challange. It is hosted here on tensorflow hub

### Custom checkpoints

For custom efficientdet models, refer to the Dockerfile for details on how to export your own model from EfficientDet

###  Building the typescript

    yarn build

#### Building the model

There is a Dockerfile in hub/ that will build an image containing an EfficientDet model and convert it to the tensorflow.js format

    cd hub
    docker build -t efficientdet-model-d0 .
    
    # Or to build a different moodel size
    # docker build --build-arg SIZE=d1 -t efficientdet-model-d0 .

    # Copy exported model files into current directory
    docker run -v (pwd):/out efficientdet-model-d0  cp -r /tmp/efficientdet-d0.js /out/
