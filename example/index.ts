import 'babel-polyfill';

import * as tf from '@tensorflow/tfjs';
import { Tensor3D } from '@tensorflow/tfjs';

import { drawBoxes } from './drawBoxes';
import { blur, getGaussianKernel } from './blur';

window.tf = tf

// tf.enableDebugMode()
tf.enableProdMode()

const size = 1200;


const blurCanvas = document.createElement("canvas")
const videoElement = <HTMLVideoElement>document.getElementById('video');
const liveCanvasElement = <HTMLCanvasElement>document.getElementById('livecanvas');
const testCanvasElement = <HTMLCanvasElement>document.getElementById('testcanvas');

const blurRadius = <HTMLInputElement>document.getElementById("blur")


var gaussianKernel = getGaussianKernel(13)

videoElement.width = videoElement.height = size;

blurRadius.addEventListener("change", function onBlurChange(e) {
  var value = parseFloat(e.target.value)
  gaussianKernel = getGaussianKernel(value)
})


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

var camera

async function capturePhoto(): Promise<Tensor3D> {
  return (await camera).capture()
}


const timeEl = document.getElementById("time")

var model
var initialized = false

async function loop() {
  console.time("loop")
  let start = Date.now()

  let img = await capturePhoto()
  let [height, width] = img.shape

  if (!initialized) {
    videoElement.height = liveCanvasElement.height = blurCanvas.height = height
    videoElement.width = liveCanvasElement.width = blurCanvas.width = width
    timeEl.style.marginTop = (height + 50) + "px"
    initialized = true
  }

  const blurredImg = blur(img, gaussianKernel)
  let batch = blurredImg.expandDims()

  // return await tf.browser.toPixels(blurredImg, liveCanvasElement)

  console.time("Prediction")
  let result = await model.executeAsync({ image_arrays: batch })
  // let result = window.pred = await model.executeAsync({ image_tensor: batch })
  console.timeEnd("Prediction")

  let detectedObjects = []
  let predictions = result.arraySync()[0] || []

  predictions.forEach(out => {
    let [image_id, y, x, ymax, xmax, score, _class] = out
    console.log(x, y)
    if (score < .05) return
    detectedObjects.push({
      bbox: {
        x: x,
        y: y,
        width: xmax - x,
        height: ymax - y
      },
      "class": labels[_class] || _class,
      score
    })
  });


  tf.dispose(blurredImg)
  tf.dispose(batch)
  tf.dispose(result)
  tf.dispose(img)

  window.requestAnimationFrame(async () => {

    drawBoxes(detectedObjects, liveCanvasElement)
    timeEl.innerText = Date.now() - start + "ms "

    window.requestAnimationFrame(loop)
  })

  console.timeEnd("loop")
}


async function test(model, imgSrc: HTMLCanvasElement | HTMLVideoElement | HTMLImageElement | ImageData) {
  let img = tf.browser.fromPixels(imgSrc)
  let [height, width] = img.shape
  testCanvasElement.height = blurCanvas.height = height
  testCanvasElement.width = blurCanvas.width = width

  const blurredImg = blur(img, gaussianKernel)
  let batch = blurredImg.expandDims()


  console.time("Prediction")
  let result = await model.executeAsync({ image_arrays: batch })
  // let result = window.pred = await model.executeAsync({ image_tensor: batch })
  console.timeEnd("Prediction")

  let detectedObjects = []
  let predictions = result.arraySync()[0]

  console.log(predictions)
  predictions.forEach(out => {
    let [image_id, y, x, ymax, xmax, score, _class] = out
    console.log(x, y)
    if (score < .05) return
    detectedObjects.push({
      bbox: {
        x: x,
        y: y,
        width: xmax - x,
        height: ymax - y
      },
      "class": labels[_class] || _class,
      score
    })
  });

  console.log("drawing")
  await tf.browser.toPixels(img, testCanvasElement)

  tf.dispose(img)
  tf.dispose(blurredImg)
  tf.dispose(batch)
  tf.dispose(result)

  drawBoxes(detectedObjects, testCanvasElement, false)
}

async function start() {
  await tf.ready()
  console.log("Initialized " tf.backend())

  // model = window.model = await tf.loadGraphModel('zach_model/model.json')


  // my model
  // model = window.model = await tf.loadGraphModel('web/model.json')

  // efficient det d0
  model = window.model = await tf.loadGraphModel('d0/model.json')

  const testJPGImg = document.getElementById("img") as HTMLImageElement

  await test(model, testJPGImg)

  document.getElementById("upload").addEventListener("change", (e) => {
    var fileInput = e.target as HTMLInputElement

    testJPGImg.onload = () => {
      test(model, testJPGImg)
    }

    testJPGImg.src = URL.createObjectURL(fileInput.files[0]);
  })

  document.getElementById("startLive").onclick = function () {
    this.hidden = true
    videoElement.hidden = false;

    (async function startLiveDemo() {
      console.log("getting camera")

      const camConfig = {
        facingMode: 'environment',
        centerCrop: false
      }

      camera = tf.data.webcam(videoElement, camConfig);
      await camera

      console.log("loop")

      loop()
    })()
  }

}



start()