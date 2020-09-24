import 'babel-polyfill';

import * as tf from '@tensorflow/tfjs';
import { Tensor3D } from '@tensorflow/tfjs';
import getDetections from "./postprocess"

import { drawBoxes } from './drawBoxes';
import { blur, getGaussianKernel } from './blur';

window.tf = tf

// tf.enableDebugMode()
tf.enableProdMode()

const size = 1200;

const camConfig = {
  facingMode: 'environment',
  centerCrop: false
}

const blurCanvas = document.createElement("canvas")
const videoElement = <HTMLVideoElement>document.getElementById('video');
const liveCanvasElement = <HTMLCanvasElement>document.getElementById('livecanvas');
const testCanvasElement = <HTMLCanvasElement>document.getElementById('testcanvas');

const blurRadius = <HTMLInputElement>document.getElementById("blur")
const blurText = <HTMLParagraphElement>document.getElementById("blurtext")


var gaussianKernel = getGaussianKernel(13)

videoElement.width = videoElement.height = size;

blurRadius.addEventListener("change", function onBlurChange(e) {
  var value = parseFloat(e.target.value)
  gaussianKernel = getGaussianKernel(value)
})

// const labels = {
//   1: 'pick',
//   2: 'red bob',
//   4: 'logo',
//   3: 'vent'
// }

const camera = tf.data.webcam(videoElement, camConfig);

async function capturePhoto(): Promise<Tensor3D> {
  let cam = await camera;
  let img = await cam.capture();
  return img as Tensor3D
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
  let result = window.pred = await model.executeAsync({ image_tensor: batch })
  console.timeEnd("Prediction")


  let detectedObjects = await getDetections(result, width, height)

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


async function test(model) {
  const testJPGImg = document.getElementById("img") as HTMLImageElement

  let img = tf.browser.fromPixels(testJPGImg)
  let [height, width] = img.shape
  testCanvasElement.height = blurCanvas.height = height
  testCanvasElement.width = blurCanvas.width = width

  const blurredImg = blur(img, gaussianKernel)
  let batch = blurredImg.expandDims()


  console.time("Prediction")
  let result = window.pred = await model.executeAsync({ image_tensor: batch })
  console.timeEnd("Prediction")


  let detectedObjects = await getDetections(result, width, height)
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

  model = window.model = await tf.loadGraphModel('zach_model/model.json')
  // let model = window.model = await tf.loadGraphModel('d0/model.json')

  await test(model)

  console.log("getting camera")
  await camera

  console.log("loop")

  loop()
}


start()