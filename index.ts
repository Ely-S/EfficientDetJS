import * as tf from '@tensorflow/tfjs';
import { Activation } from '@tensorflow/tfjs-layers/dist/activations';

import "babel-polyfill"
import { Dropout } from '@tensorflow/tfjs-layers/dist/layers/core';


export class swish extends Activation {
    /** @nocollapse */
    static readonly className = 'swish';
    /**
     * Calculate the activation function.
     *
     * @param x: Input.
     * @param alpha: Scaling factor for the sigmoid function.
     * @return Output of the Swish activation.
     */
    apply(x: tf.Tensor, alpha = 1): tf.Tensor {
        return tf.sigmoid(x.mul(alpha)).mul(x);
    }
}

tf.serialization.registerClass(swish);

class FixedDropout extends Dropout {
    static className = 'FixedDropout';
}

tf.serialization.registerClass(FixedDropout);

// There is an operation in EfficientDet which uses a 
// version of keras.dropout with a bug fixed. It is called FixedDropout

const camConfig = {
    // facingMode: 'user', //'environment'
    resizeWidth: 416,
    resizeHeight: 416,
    centerCrop: true
}

const videoElement = <HTMLVideoElement> document.getElementById('video');
const canvasElement = <HTMLCanvasElement>  document.getElementById('canvas');

videoElement.width = 640;
videoElement.height = 480;

canvasElement.width = 512;
canvasElement.height = 512;

const camera = tf.data.webcam(videoElement, camConfig);

async function capturePhoto(){
    let cam = await camera;
    let img = await cam.capture();

    let scaledImage = img.div(tf.scalar(255))
    tf.browser.toPixels(scaledImage, canvasElement) 
    return scaledImage
}

async function startLoop(model){
    let image = await capturePhoto()
    model.predict(image)
}

capturePhoto()

class Lambda extends tf.layers.Layer {
    constructor() {
      super({})
    }
  
    static get className() {
      return 'Lambda';
    }
  
     call(inputs, kwargs) {
      let input = inputs;
      if (Array.isArray(input)) {
        input = input[0];
      }
      this.invokeCallHook(inputs, kwargs);
      /*const origShape = input.shape;
      const flatShape =
          [origShape[0], origShape[1] * origShape[2] * origShape[3]];
      const flattened = input.reshape(flatShape);
      const centered = tf.sub(flattened, flattened.mean(1).expandDims(1));
      const pos = centered.relu().reshape(origShape);
      const neg = centered.neg().relu().reshape(origShape);
      return tf.concat([pos, neg], 3); */
      return input.pow(tf.tensor(2).toInt())
    }
}
  
tf.serialization.SerializationMap.register(Lambda);

const loadModelPromise =  tf.loadLayersModel('/model2/model.json');

loadModelPromise.then(model => {
    model.summary()
    startLoop(model)
})

// const example = tf.fromPixels(webcamElement);  // for example
// const prediction = model.predict(example);


document.getElementById("hi").innerText = 'hi me'