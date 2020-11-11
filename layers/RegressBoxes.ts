import * as tf from "@tensorflow/tfjs";
import { LayerArgs } from "@tensorflow/tfjs-layers/src/engine/topology";
import { Tensor1D, Shape, Tensor } from "@tensorflow/tfjs";

import * as hacks from "./hacks";

hacks.init(tf.Tensor)

interface RegressBoxesArgs extends LayerArgs {
    // standard deviation to normalize by
    std?: Array<number> | Tensor1D;
    mean?: Array<number> | Tensor1D;
    anchorShape?: Shape;
}

const default_mean = tf.tensor1d([0, 0, 0, 0], 'float32')
const default_std = tf.tensor1d([0.2, 0.2, 0.2, 0.2], 'float32')

function parseArgument(arg: Tensor | Array<number>, defaultValue: Tensor1D): Tensor1D {
    if (arg === undefined || arg === null) return defaultValue
    if (arg instanceof Array) return tf.tensor1d(arg, 'float32')
    if (arg instanceof tf.Tensor) {
        if (arg.shape.length !== 1) throw Error("Expected 1d Tensor, recieved" + arg)
        return arg as Tensor1D
    }

    console.error("Argument", arg, "is not an Array or Tensor")
    throw Error("Argument " + arg + "is not an Array or Tensor")
}

export class RegressBoxes extends tf.layers.Layer {
    static className = 'RegressBoxes';

    // @TODO: see what happens when this is true
    private std: Tensor1D
    private mean: Tensor1D

    // This is anchor_shape is python
    // The naming convention gets mangled by conversion
    private anchorShape: Shape

    // weights
    private anchors: tf.LayerVariable = null

    constructor(args: RegressBoxesArgs) {
        super(args)

        this.setFastWeightInitDuringBuild(true)

        this.std = parseArgument(args.std, default_std)
        this.mean = parseArgument(args.mean, default_mean)
        this.anchorShape = args.anchorShape
        this.build()
    }

    public build(): void {
        this.anchors = this.addWeight(
            "anchor_boxes_baked", this.anchorShape, "float32",
            undefined, undefined, false // untrainable
        )
        this.built = true;
    }

    call(inputs: Tensor, kwargs) {
        return tf.tidy(() => {
            this.invokeCallHook(inputs, kwargs);

            let tensorsTensor = this.anchors.read()
            return apply_bbox_deltas(
                tensorsTensor as hacks.EasyTensor,
                inputs[0] as hacks.EasyTensor,
                this.mean, this.std)
        });
    }

    computeOutputShape(): Shape {
        return this.anchorShape
    }

    getConfig(): tf.serialization.ConfigDict {
        const config = super.getConfig();

        config.mean = this.mean.arraySync()
        config.std = this.std.arraySync()
        config.anchor_shape = this.anchorShape

        return config;
    }

    getClassName() {
        return 'RegressBoxes';
    }

}

export function apply_bbox_deltas(
    boxes: hacks.EasyTensor,
    deltas: hacks.EasyTensor,
    mean = default_mean,
    std = default_std) {
    /*
    Applies deltas (usually regression results) to boxes (usually anchors).

    Before applying the deltas to the boxes, the normalization that was
    previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator.
    They are unnormalized in this function and then applied to the boxes.

    :param np.array boxes: np.array of shape (B, N, 4), where B is the batch
         size, N the number of boxes and 4 values for (x1, y1, x2, y2).
    :param np.array deltas: np.array of same shape as boxes.
        These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
    :param list mean: The mean value used when computing deltas
        (defaults to [0, 0, 0, 0]).
    :param list std: The standard deviation used when computing deltas

    :return: A np.array of the same shape as boxes, but with deltas applied to
        each box. The mean and std are used during training to normalize the
        regression values (networks love normalization).
    :rtype: np.array
    */

    let widths = boxes.$(':, :, 2').sub(boxes.$(':, :, 0'))
    let heights = boxes.$(':, :, 3').sub(boxes.$(':, :, 1'))

    let scales = tf.stack([widths, heights, widths, heights], 2)

    let normalized_deltas = deltas.$(':, :').mul(std).add(mean)
    let bboxes = boxes.$(':, :').add(normalized_deltas.mul(scales))

    return bboxes
}
