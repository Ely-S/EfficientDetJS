import * as tf from '@tensorflow/tfjs';

import "babel-polyfill"
import { Tensor3D } from '@tensorflow/tfjs';
import { Swish } from './operations';
import {  RegressBoxes } from './layers/RegressBoxes'
import { Reshape } from './layers/Reshape';

tf.enableDebugMode()
tf.setBackend("webgl")

tf.serialization.registerClass(Swish);
tf.serialization.registerClass(RegressBoxes);
tf.serialization.registerClass(Reshape);
  

const camConfig = {
    // facingMode: 'user',
    resizeWidth: 512,
    resizeHeight: 512,
    centerCrop: true
}

const videoElement = <HTMLVideoElement> document.getElementById('video');
const canvasElement = <HTMLCanvasElement>  document.getElementById('canvas');

videoElement.width = 640;
videoElement.height = 480;

canvasElement.width = 512;
canvasElement.height = 512;

const camera = tf.data.webcam(videoElement, camConfig);

window.tf = tf

async function capturePhoto(){
    let cam = await camera;
    let img = await cam.capture();

    let scaledImage = img.div(tf.scalar(255)) as Tensor3D
    tf.browser.toPixels(scaledImage, canvasElement)
}


capturePhoto()


async function start() {
    await tf.ready()
    let model = await tf.loadLayersModel('/unweighted_nodp/model.json')
    console.log(model)    
}

start()