import 'babel-polyfill';

import * as tf from '@tensorflow/tfjs';
import {Tensor3D} from '@tensorflow/tfjs';

import {PriorProbability} from './initializers';
import {ClipBoxes} from './layers/ClipBoxes';
import {FilterDetections} from './layers/FilterDetections';
import {RegressBoxes} from './layers/RegressBoxes';
import {Reshape} from './layers/Reshape';
import {SigmoidLayer, Swish, SwishLayer} from './layers/Sigmoids';

window.tf = tf

tf.enableProdMode()
tf.setBackend('webgl')

tf.serialization.registerClass(Swish);
tf.serialization.registerClass(RegressBoxes);
tf.serialization.registerClass(Reshape);
tf.serialization.registerClass(PriorProbability)
tf.serialization.registerClass(ClipBoxes)
tf.serialization.registerClass(FilterDetections)
tf.serialization.registerClass(SwishLayer)
tf.serialization.registerClass(SigmoidLayer)

const camConfig = {
  // facingMode: 'user',
  resizeWidth: 640,
  resizeHeight: 640,
  centerCrop: true
}

const videoElement = <HTMLVideoElement>document.getElementById('video');
const canvasElement = <HTMLCanvasElement>document.getElementById('canvas');

videoElement.width = 640;
videoElement.height = 480;

canvasElement.width = 640;
canvasElement.height = 640;

const camera = tf.data.webcam(videoElement, camConfig);

async function capturePhoto() {
  let cam = await camera;
  let img = await cam.capture();

  let scaledImage = img.div(tf.scalar(255)) as Tensor3D
  tf.browser.toPixels(scaledImage, canvasElement)
}


capturePhoto()


async function start() {
  await tf.ready()
  let model = await tf.loadLayersModel('/pascal_phi1_unweighted/model.json')
  // '/pascal_unweighted_sigmoidlayer_swishlayer_nofilter/model.json')

  window.model = model

  model.summary()

  let cam = await camera;
  let img = await cam.capture();

  let scaledImage = img.div(tf.scalar(255)) as Tensor3D
  let batch = scaledImage.expandDims()

  let dumb_anchors = tf.ones([1, 1, 4]).mul(.5)

  console.log('DONE LOADING')
  // https://github.com/tensorflow/tfjs/blob/fe4627f11effdff3b329920eae57a4c4b1e4c67c/tfjs-core/src/util.ts#L423
  model.predict([batch, dumb_anchors], {verbose: true})

  console.log('done predicting')

  for(let i = 0; i < 100; i++) {
    console.time("Prediction")
    model.predict([batch, dumb_anchors])
    console.timeEnd("Prediction")  
  }

}

start()