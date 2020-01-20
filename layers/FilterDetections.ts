import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from "@tensorflow/tfjs-layers/src/engine/topology";
import { Tensor, NamedTensorInfoMap, NamedTensorMap } from '@tensorflow/tfjs';

interface filterDetectionArgs extends LayerArgs {
  nms: boolean,
  soft: boolean,
  iouThreshold: number,
  softNmsSigma?: number,
  maxOutputSize: number,
  scoreThreshold: number,
}

export class FilterDetections extends tf.layers.Layer {
  /** @nocollapse */
  static className = 'FilterDetections';

  nmsFunc(boxes: Tensor<Rank>, scores: Tensor<Rank>): Promise<NamedTensorMap>

  constructor(args: filterDetectionArgs) {
    super(args)

    let { maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma } = args

    if (args.soft) {
      this.nmsFunc = (boxes, scores) => {
        let [selected_indeces, selected_scores] = tf.image.nonMaxSuppressionWithScoreAsync(
          boxes, scores, maxOutputSize, iouThreshold,
          scoreThreshold, softNmsSigma)
        return [selected_indeces, selected_scores]
      }
    } else {
      this.nmsFunc = (boxes, scores) => {
        return tf.image.nonMaxSuppression(boxes, scores,
          maxOutputSize, iouThreshold, scoreThreshold)
      }

    }
  }

  call() {

  }
}
