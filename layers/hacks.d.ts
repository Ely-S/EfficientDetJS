import {Tensor} from '@tensorflow/tfjs';

declare interface EasyTensor extends Tensor {
  $(String): EasyTensor
}

export function init(Class): EasyTensor
