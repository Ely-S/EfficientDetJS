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

tf.enableDebugMode()
tf.setBackend('cpu')

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
  resizeWidth: 512,
  resizeHeight: 512,
  centerCrop: true
}

const videoElement = <HTMLVideoElement>document.getElementById('video');
const canvasElement = <HTMLCanvasElement>document.getElementById('canvas');

videoElement.width = 640;
videoElement.height = 480;

canvasElement.width = 512;
canvasElement.height = 512;

const camera = tf.data.webcam(videoElement, camConfig);

async function capturePhoto() {
  let cam = await camera;
  let img = await cam.capture();

  let scaledImage = img.div(tf.scalar(255)) as Tensor3D
  tf.browser.toPixels(scaledImage, canvasElement)
}


// capturePhoto()


async function start() {
  await tf.ready()
  let model = await tf.loadLayersModel(
      '/pascal_unweighted_sigmoidlayer_swishlayer_nofilter/model.json')

  model.summary()
}

start()