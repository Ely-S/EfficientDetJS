import {Dropout} from '@tensorflow/tfjs-layers/dist/layers/core';


export class FixedDropout extends Dropout {
  // There is an operation in EfficientDet which uses a
  // version of keras.dropout with a bug fixed. It is called FixedDropout.
  // This is hoping that the bug is fixed in tf.js.
  // This is unnecessary if dropout layers are removed at the export step.
  static className = 'FixedDropout';

  getClassName() {
    return 'FixedDropout'
  }

  constructor(args) {
    super(args)
  }
}