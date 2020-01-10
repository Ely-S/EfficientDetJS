import { Shape, Tensor, tidy, layers, serialization } from '@tensorflow/tfjs';
import { ReshapeLayerArgs } from '@tensorflow/tfjs-layers/src/layers/core';
import { getExactlyOneTensor } from '@tensorflow/tfjs-layers/src/utils/types_utils';
import { Kwargs } from '@tensorflow/tfjs-layers/dist/types';
import { ValueError } from '@tensorflow/tfjs-layers/src/errors';
import { arrayProd } from '@tensorflow/tfjs-layers/src/utils/math_utils';

// This was coppied directly out of tensorflow. 
// Registering it again makes it work for some reason.
export class Reshape extends layers.Layer {
    /** @nocollapse */
    static className = 'Reshape';
    private targetShape: Shape;
    constructor(args: ReshapeLayerArgs) {
        super(args);
        this.targetShape = args.targetShape;
        // Make sure that all unknown dimensions are represented as `null`.
        for (let i = 0; i < this.targetShape.length; ++i) {
            if (this.isUnknown(this.targetShape[i])) {
                this.targetShape[i] = null;
            }
        }
    }

    private isUnknown(dim: number): boolean {
        return dim < 0 || dim == null;
    }

    /**
     * Finds and replaces a missing dimension in output shape.
     *
     * This is a near direct port of the internal Numpy function
     * `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`.
     *
     * @param inputShape: Original shape of array begin reshape.
     * @param outputShape: Target shape of the array, with at most a single
     * `null` or negative number, which indicates an underdetermined dimension
     * that should be derived from `inputShape` and the known dimensions of
     *   `outputShape`.
     * @returns: The output shape with `null` replaced with its computed value.
     * @throws: ValueError: If `inputShape` and `outputShape` do not match.
     */
    private fixUnknownDimension(inputShape: Shape, outputShape: Shape): Shape {
        const errorMsg = 'Total size of new array must be unchanged.';
        const finalShape = outputShape.slice();
        let known = 1;
        let unknown = null;
        for (let i = 0; i < finalShape.length; ++i) {
            const dim = finalShape[i];
            if (this.isUnknown(dim)) {
                if (unknown === null) {
                    unknown = i;
                }
                else {
                    throw new ValueError('Can only specifiy one unknown dimension.');
                }
            }
            else {
                known *= dim;
            }
        }
        const originalSize = arrayProd(inputShape);
        if (unknown !== null) {
            if (known === 0 || originalSize % known !== 0) {
                throw new ValueError(errorMsg);
            }
            finalShape[unknown] = originalSize / known;
        }
        else if (originalSize !== known) {
            throw new ValueError(errorMsg);
        }
        return finalShape;
    }

    computeOutputShape(inputShape: Shape): Shape {
        let anyUnknownDims = false;
        for (let i = 0; i < inputShape.length; ++i) {
            if (this.isUnknown(inputShape[i])) {
                anyUnknownDims = true;
                break;
            }
        }
        if (anyUnknownDims) {
            return inputShape.slice(0, 1).concat(this.targetShape);
        }
        else {
            return inputShape.slice(0, 1).concat(this.fixUnknownDimension(inputShape.slice(1), this.targetShape));
        }
    }
    
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[] {
        console.log(inputs);
        return tidy(() => {
            console.log(inputs, 1);
            this.invokeCallHook(inputs, kwargs);
            const input = getExactlyOneTensor(inputs);
            const inputShape = input.shape;
            const outputShape = inputShape.slice(0, 1).concat(this.fixUnknownDimension(inputShape.slice(1), this.targetShape));
            return input.reshape(outputShape);
        });
    }

    getConfig(): serialization.ConfigDict {
        const config = {
            targetShape: this.targetShape,
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
