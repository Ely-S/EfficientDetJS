import * as tf from '@tensorflow/tfjs';
const MIN_SCORE = 0.09
const MAX_PREDICTIONS = 1
const IOU_THRESHOLD = .5

export default async function getDetections(result, width, height) {
    console.time("GET DETECTION")

    // zachs model has output detection_Scores, num_detections, detection_boxes 
    var [scores, boxes] = await Promise.all([result[0].data(), result[1].data()]) as Float32Array[];

    const prevBackend = tf.getBackend();
    // run post process in cpu
    tf.setBackend('cpu');

    // The score for every class is predicted for each box
    // Get the maximum class score and class for each box
    const [maxScores, classes] = calculateMaxScores(scores, result[0].shape[1], result[0].shape[2]);

    console.time("NMS")
    const indexes = tf.tidy(() => {
        const boxes2 =
            tf.tensor2d(boxes, [result[1].shape[1], result[1].shape[3]]);

        return tf.image.nonMaxSuppression(
            boxes2, maxScores, MAX_PREDICTIONS, IOU_THRESHOLD, MIN_SCORE
        ).dataSync()
    }) as Float32Array;
    console.timeEnd("NMS")

    let detectedObjects = buildDetectedObjects(
        width, height, boxes, maxScores, indexes, classes);

    tf.setBackend(prevBackend);

    console.log(detectedObjects)

    console.timeEnd("GET DETECTION")


    return detectedObjects
}

interface DetectedObject {
    bbox: [number, number, number, number];  // [x, y, width, height]
    class: string;
    score: number;
}


function buildDetectedObjects(
    width: number, height: number, boxes: Float32Array, scores: number[],
    indexes: Float32Array, classes: number[]): DetectedObject[] {
    const count = indexes.length;
    const objects: DetectedObject[] = [];
    for (let i = 0; i < count; i++) {
        const bbox = [];
        for (let j = 0; j < 4; j++) {
            bbox[j] = boxes[indexes[i] * 4 + j];
        }
        const minY = bbox[0] * height;
        const minX = bbox[1] * width;
        const maxY = bbox[2] * height;
        const maxX = bbox[3] * width;
        bbox[0] = minX;
        bbox[1] = minY;
        bbox[2] = maxX - minX;
        bbox[3] = maxY - minY;
        objects.push({
            bbox: bbox as [number, number, number, number],
            class: "dozer",
            score: scores[indexes[i]]
        });
    }
    return objects;
}

function calculateMaxScores(
    scores: Float32Array, numBoxes: number,
    numClasses: number): [number[], number[]] {

    const maxes = [];
    const classes = [];
    for (let i = 0; i < numBoxes; i++) {
        let max = Number.MIN_VALUE;
        let index = -1;
        for (let j = 0; j < numClasses; j++) {
            if (scores[i * numClasses + j] > max) {
                max = scores[i * numClasses + j];
                index = j;
            }
        }
        maxes[i] = max;
        classes[i] = index;
    }
    return [maxes, classes];
}
