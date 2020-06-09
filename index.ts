import 'babel-polyfill';

import * as tf from '@tensorflow/tfjs';
import { Tensor3D } from '@tensorflow/tfjs';

import { drawBoxes } from './drawBoxes';

window.tf = tf

tf.enableProdMode()
tf.setBackend('webgl')

const targetCanvas = document.createElement("canvas")
targetCanvas.height = targetCanvas.width = 512

const size = 512;

const camConfig = {
  facingMode: 'user',
  resizeWidth: size,
  resizeHeight: size,
  centerCrop: false
}

const videoElement = <HTMLVideoElement>document.getElementById('video');
const liveCanvasElement = <HTMLCanvasElement>document.getElementById('canvas');

// Labels for the COCO dataset
const labels = {
  1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
  6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
  11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
  16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
  22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
  28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
  35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
  39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
  43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
  49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
  54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
  59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
  64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
  73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
  78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
  84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
  89: 'hair drier', 90: 'toothbrush',
}
const camera = tf.data.webcam(videoElement, camConfig);

async function capturePhoto(): Promise<Tensor3D> {
  let cam = await camera;
  let img = await cam.capture();
  return img as Tensor3D
}


const timeEl = document.getElementById("time")

async function loop(model) {
  let start = Date.now()

  let img = await capturePhoto()
  let originalImg = img

  await tf.browser.toPixels(img, targetCanvas)

  img = tf.browser.fromPixels(targetCanvas)
  let scaledImage = tf.cast(img, "int32") as Tensor3D
  let batch = scaledImage.expandDims()

  console.time("Prediction")
  let p = await model.executeAsync({ image_arrays: batch })
  console.timeEnd("Prediction")

  let boxes = []
  let predictions = p.arraySync()[0]

  predictions.forEach(out => {
    let [image_id, y, x, ymax, xmax, score, _class] = out
    if (score < .1) return
    boxes.push({
      bbox: {
        x: x,
        y: y,
        width: xmax - x,
        height: ymax - y
      },
      "class": labels[_class],
      score
    })
  });

  await tf.browser.toPixels(originalImg, liveCanvasElement)

  drawBoxes(boxes, liveCanvasElement)

  timeEl.innerText = Date.now() - start + "ms"
  loop(model)
}

async function start() {
  await tf.ready()

  let model = await tf.loadGraphModel('d0/model.json')
  window.model = model // to explore in the console

  await loop(model)
}


start()