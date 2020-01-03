import * as tf from '@tensorflow/tfjs';

import "babel-polyfill"


const camConfig = {
    facingMode: 'user', //'environment'
    resizeWidth: 512,
    resizeHeight: 512,
    centerCrop: true
}

const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');

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
}


capturePhoto()


const loadModelPromise =  tf.loadLayersModel('/model/model.json');



// const example = tf.fromPixels(webcamElement);  // for example
// const prediction = model.predict(example);


document.getElementById("hi").innerText = 'hi me'