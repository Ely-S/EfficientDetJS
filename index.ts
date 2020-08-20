import 'babel-polyfill';

import * as tf from '@tensorflow/tfjs';
import { Tensor3D } from '@tensorflow/tfjs';

import { canvasRGBA } from "stackblur-canvas"

import * as hacks from "./layers/hacks";

hacks.init(tf.Tensor)
const radius = 4


// import { PriorProbability } from './initializers';
// import { ClipBoxes } from './layers/ClipBoxes';
// import { FilterDetections } from './layers/FilterDetections';
// import { RegressBoxes } from './layers/RegressBoxes';
// import { Swish } from './layers/Sigmoids';
// import { wBiFPNAdd } from './layers/wBiFPNAddLayer';
import { drawBoxes } from './drawBoxes';

window.tf = tf

// tf.enableDebugMode()
tf.enableProdMode()
tf.setBackend('webgl')

// tf.serialization.registerClass(Swish);
// tf.serialization.registerClass(RegressBoxes);
// tf.serialization.registerClass(wBiFPNAdd);
// tf.serialization.registerClass(PriorProbability)
// tf.serialization.registerClass(ClipBoxes)
// tf.serialization.registerClass(FilterDetections)
const targetCanvas = document.createElement("canvas")
targetCanvas.height = targetCanvas.width = 300

const size = 300;

const camConfig = {
  facingMode: 'user',
  centerCrop: false
}

const videoElement = <HTMLVideoElement>document.getElementById('video');
const liveCanvasElement = <HTMLCanvasElement>document.getElementById('canvas');
const testCanvasElement = <HTMLCanvasElement>document.getElementById('testcanvas');

testCanvasElement.width = liveCanvasElement.width = size;
testCanvasElement.height = liveCanvasElement.height = size;
videoElement.width = videoElement.width = size;

// const labels = {
//   1: 'pick',
//   2: 'red bob',
//   4: 'logo',
//   3: 'vent'
// }

// const labels = {
//   1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
//   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
//   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
//   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
//   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
//   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
//   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
//   39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
//   43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
//   49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
//   54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
//   59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
//   64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
//   73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
//   78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
//   84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
//   89: 'hair drier', 90: 'toothbrush',
// }
const camera = tf.data.webcam(videoElement, camConfig);

async function capturePhoto(): Promise<Tensor3D> {
  let cam = await camera;
  let img = await cam.capture();
  return img as Tensor3D
}


const timeEl = document.getElementById("time")

async function loop(model) {
  let start = Date.now()

  let rawImg = await capturePhoto() as Tensor3D

  let img = tf.image.resizeNearestNeighbor(
    rawImg, [300, 300], false)

  let originalImg = img

  await tf.browser.toPixels(img.div(255), liveCanvasElement)

  canvasRGBA(targetCanvas, 0, 0, 300, 300, 4);

  img = tf.browser.fromPixels(targetCanvas)

  // https://github.com/tensorflow/tfjs/blob/fe4627f11effdff3b329920eae57a4c4b1e4c67c/tfjs-core/src/util.ts#L423
  let scaledImage = tf.cast(img, "int32") as Tensor3D

  let batch = scaledImage.expandDims()

  console.time("Prediction")
  // effficient-det model
  // let p = window.pred = await model.executeAsync({ image_arrays: batch })

  // zach model
  let p = window.pred = await model.executeAsync({ image_tensor: batch })

  console.timeEnd("Prediction")
  let [detection_classes, num_detections, detection_boxes, detection_scores] = p

  detection_scores = detection_scores.dataSync()
  num_detections = num_detections.dataSync()
  detection_boxes = detection_boxes.dataSync()

  let boxes = []

  detection_scores.forEach((score, index) => {
    let [y, x, ymax, xmax] = detection_boxes
    console.log(score)
    // let [image_id, y, x, ymax, xmax, score, _class] = out
    if (score < .01) return
    boxes.push({
      bbox: {
        x: x,
        y: y,
        width: xmax - x,
        height: ymax - y
      },
      "class": "bulldozer",
      score
    })
  });

  await tf.browser.toPixels(originalImg.div(255), liveCanvasElement)

  drawBoxes(boxes, liveCanvasElement)

  timeEl.innerText = Date.now() - start + "ms " + detection_scores.join(" - ")
  loop(model)
}


async function test(model) {
  // @TODO: This needs to use tf.tidy
  const testJPGCanvas = <HTMLImageElement>document.getElementById("img")

  let img = tf.image.resizeNearestNeighbor(
    tf.browser.fromPixels(testJPGCanvas), [300, 300], true)
  let originalImg = img

  await tf.browser.toPixels(img.div(255), targetCanvas)

  canvasRGBA(targetCanvas, 0, 0, 300, 300, 2);
  img = tf.browser.fromPixels(targetCanvas)

  let scaledImage = tf.cast(img, "int32") as Tensor3D

  let batch = scaledImage.expandDims()

  // https://github.com/tensorflow/tfjs/blob/fe4627f11effdff3b329920eae57a4c4b1e4c67c/tfjs-core/src/util.ts#L423

  console.time("Prediction")
  let pred = window.pred = await model.executeAsync({ image_tensor: batch })
  console.timeEnd("Prediction")

  // zach
  // zachs model has output detection_Scores, num_detections, detection_boxes 
  let [detection_classes, num_detections, detection_boxes, detection_scores] = pred

  detection_scores = detection_scores.dataSync()
  num_detections = num_detections.dataSync()
  detection_boxes = detection_boxes.dataSync()

  // efficientdet
  // p has shape [batch_size, detections, predictions]

  // let predScores = p.$(":,:,5").dataSync()
  // let predBoxes = p.$(":,:,1:5").dataSync()
  // let predClasses = p.$(":,:,6").dataSync()

  await tf.browser.toPixels(originalImg.div(255), testCanvasElement)

  let boxes = []

  detection_scores.forEach((score, index) => {
    let [y, x, ymax, xmax] = detection_boxes
    if (score < .01) return
    boxes.push({
      bbox: {
        x: x * 300,
        y: y * 300,
        width: (xmax - x) * 300,
        height: (ymax - y) * 300
      },
      "class": "bulldozer",
      score
    })
  });

  console.log(boxes)
  drawBoxes(boxes, testCanvasElement)
}


async function start() {
  await tf.ready()

  let model = window.model = await tf.loadGraphModel('zach_model/model.json')
  // let model = window.model = await tf.loadGraphModel('d0/model.json')

  await test(model)
  await loop(model)
}


start()