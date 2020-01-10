import * as tf from '@tensorflow/tfjs';
import { Activation } from '@tensorflow/tfjs-layers/dist/activations';

import { Dropout } from '@tensorflow/tfjs-layers/dist/layers/core';

export class Swish extends Activation {
    /** @nocollapse */
    static readonly className = 'swish';
    /**
     * Calculate the activation function.
     *
     * @param x: Input.
     * @param alpha: Scaling factor for the sigmoid function.
     * @return Output of the Swish activation.
     */
    apply(x: tf.Tensor, alpha = 1): tf.Tensor {
        return tf.sigmoid(x.mul(alpha)).mul(x);
    }
}

export class FixedDropout extends Dropout {
    // There is an operation in EfficientDet which uses a 
    // version of keras.dropout with a bug fixed. It is called FixedDropout.
    // This is hoping that the bug is fixed in tf.js.
    // This is unnecessary if dropout layers are removed at the export step.

    static className = 'FixedDropout';
    getClassName() { return 'FixedDropout'}
    constructor(args) {
        console.log(args)
        super(args)
    }

}