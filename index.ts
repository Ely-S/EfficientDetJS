import 'babel-polyfill';

import * as tf from '@tensorflow/tfjs';
import { Tensor3D } from '@tensorflow/tfjs';

import { PriorProbability } from './initializers';
import { ClipBoxes } from './layers/ClipBoxes';
import { FilterDetections } from './layers/FilterDetections';
import { RegressBoxes } from './layers/RegressBoxes';
import { Swish } from './layers/Sigmoids';
import { wBiFPNAdd } from './layers/wBiFPNAddLayer';
import { drawBoxes } from './drawBoxes';

window.tf = tf

// tf.enableDebugMode()
tf.enableProdMode()
tf.setBackend('webgl')

tf.serialization.registerClass(Swish);
tf.serialization.registerClass(RegressBoxes);
tf.serialization.registerClass(wBiFPNAdd);
tf.serialization.registerClass(PriorProbability)
tf.serialization.registerClass(ClipBoxes)
tf.serialization.registerClass(FilterDetections)

const size = 512;

const camConfig = {
  // facingMode: 'user',
  resizeWidth: size,
  resizeHeight: size,
  centerCrop: true
}

const videoElement = <HTMLVideoElement>document.getElementById('video');
const canvasElement = <HTMLCanvasElement>document.getElementById('canvas');

videoElement.width = 640;
videoElement.height = 480;

canvasElement.width = size;
canvasElement.height = size;

const camera = tf.data.webcam(videoElement, camConfig);

async function capturePhoto(): Promise<Tensor3D> {
  let cam = await camera;
  let img = await cam.capture();

  let scaledImage = img.div(tf.scalar(255)) as Tensor3D

  return scaledImage
}

async function loop(model) {
  // @TODO: This needs to use tf.tidy

  let scaledImage = await capturePhoto() as Tensor3D
  let batch = scaledImage.expandDims()

  // https://github.com/tensorflow/tfjs/blob/fe4627f11effdff3b329920eae57a4c4b1e4c67c/tfjs-core/src/util.ts#L423

  console.time("Prediction")
  let p = model.predict([batch])
  console.timeEnd("Prediction")

  let allBoxes = p[0]
  let allScores = p[1]

  // allBoxes has shape (batch_size, boxes, 4)
  // allScores has shape (batch_size, boxes, n_classes)
  // where boxes = n_ancor_boxes

  // perform inference on one image at a time

  let boxes = allBoxes.gather(0).squeeze()
  let scores = allScores.gather(0).squeeze()


  let maxOutPutSize = 2
  let iouThreshold = .5
  let scoreThreshold = .94


  // person is class 14
  console.time("nms")

  let scores_for_class_i = scores.$(":," + 14)

  let indecies = await tf.image.nonMaxSuppressionAsync(
    boxes, scores_for_class_i,
    maxOutPutSize, iouThreshold, scoreThreshold
  )

  let [predBoxes, predScores] = await Promise.all([
    boxes.gather(indecies).data(),
    scores_for_class_i.gather(indecies).data()
  ])

  console.timeEnd("nms")

  tf.browser.toPixels(scaledImage, canvasElement)

  for (let i = 0; i < predScores.length; i++) {
    drawBoxes([{
      bbox: predBoxes.slice(i * 4, i * 4 + 4),
      class: "human",
      score: predScores[i]
    }], canvasElement)
  }

  console.log("draw end")
  return loop(model)
}

async function start() {
  await tf.ready()

  let model = await tf.loadLayersModel('/pascal_phi0_weighted/model.json')

  // let model = await tf.loadLayersModel('/pascal_phi1_unweighted/model.json')

  window.model = model
  model.summary()

  loop(model)
}


start()