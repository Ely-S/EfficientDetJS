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

const size = 256;

const camConfig = {
  facingMode: 'user',
  resizeWidth: size,
  resizeHeight: size,
  centerCrop: false
}

const videoElement = <HTMLVideoElement>document.getElementById('video');
const canvasElement = <HTMLCanvasElement>document.getElementById('canvas');
const testCanvasElement = <HTMLCanvasElement>document.getElementById('testcanvas');

testCanvasElement.width = canvasElement.width = size;
testCanvasElement.height = canvasElement.height = size;

const labels = {
  15: "person",
  2: "bike"
}

const camera = tf.data.webcam(videoElement, camConfig);

async function capturePhoto(): Promise<Tensor3D> {
  let cam = await camera;
  let img = await cam.capture();

  let scaledImage = img.div(tf.scalar(255)) as Tensor3D

  return scaledImage
}


const timeEl = document.getElementById("time")

async function loop(model) {
  let start = Date.now()

  let scaledImage = await capturePhoto()

  // let scaledImage = await capturePhoto() as Tensor3D
  let batch = scaledImage.expandDims()

  // https://github.com/tensorflow/tfjs/blob/fe4627f11effdff3b329920eae57a4c4b1e4c67c/tfjs-core/src/util.ts#L423

  console.time("Prediction")
  let p = await model.executeAsync([batch])
  console.timeEnd("Prediction")

  // p has shape [batch_size, detections, predictions]
  // detections contains [image_id, y, x, height, width, score, class]


  let [predScores, predBoxes, predClasses, _] = await Promise.all([
    p.$(":,:,5").data(),
    p.$(":,:,1:5").data(),
    p.$(":,:,6").data(),
    tf.browser.toPixels(scaledImage, canvasElement)
  ])

  let boxes = []

  for (let i = 0; i < predScores.length; i++) {
    let bbox = predBoxes.slice(i * 4, i * 4 + 4)
    let [y, x, height, width] = bbox

    if (predScores[i] < .7) continue
    if (labels[predClasses[i]] === undefined) continue

    boxes.push({
      bbox: [y, x, height, width],
      class: labels[predClasses[i]],
      score: predScores[i]
    })
  }

  drawBoxes(boxes, canvasElement)

  timeEl.innerText = Date.now() - start + "ms"
  loop(model)
}

async function test(model) {
  // @TODO: This needs to use tf.tidy

  const image = <HTMLImageElement>document.getElementById("img")
  let img = tf.browser.fromPixels(image)
  img = tf.image.resizeBilinear(img, [size, size])

  let scaledImage = img.div(tf.scalar(255)) as Tensor3D

  let batch = scaledImage.expandDims()

  // https://github.com/tensorflow/tfjs/blob/fe4627f11effdff3b329920eae57a4c4b1e4c67c/tfjs-core/src/util.ts#L423

  console.time("Prediction")
  let p = await model.executeAsync([batch])
  console.timeEnd("Prediction")

  // p has shape [batch_size, detections, predictions]
  // detections contains [image_id, y, x, height, width, score, class]

  let predScores = tf.mul(2, p.$(":,:,5")).dataSync()
  let predBoxes = p.$(":,:,1:5").dataSync()
  let predClasses = p.$(":,:,6").dataSync()


  await tf.browser.toPixels(scaledImage, testCanvasElement)

  let boxes = []

  for (let i = 0; i < predScores.length; i++) {
    let bbox = predBoxes.slice(i * 4, i * 4 + 4)
    let [y, x, height, width] = bbox

    boxes.push({
      bbox: [x, y, x + width, y + height],
      class: labels[predClasses[i]],
      score: predScores[i]
    })
  }

  drawBoxes(boxes, testCanvasElement)
}


async function start() {
  await tf.ready()

  let model = await tf.loadGraphModel('/web/model.json')

  window.model = model

  await test(model)
  await loop(model)

}


start()