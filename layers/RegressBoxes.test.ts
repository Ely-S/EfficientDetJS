import { RegressBoxes, apply_bbox_deltas } from "./RegressBoxes"

import * as tf from "@tensorflow/tfjs";
import * as hacks from "./hacks";
import { assertTensorsEqual } from './testing'

test('tf $', () => {
    hacks.init(tf.Tensor)
    let t = tf.tensor([[1, 2], [3, 4]])
    let t2 = t.slice([0, 0], [-1, 1])
    let t3 = t.$(":,:1")
    assertTensorsEqual(t2, t3)
});


describe("RegressBoxes Layer", () => {
    hacks.init(tf.Tensor)

    let boxes = tf.tensor(
        [ // batch of predictions
            [ // list of boxes
                // list of 4 coords
                [0, 0, 1, 1],
                [.5, .5, .6, .6]
            ]
        ]
    ) as hacks.EasyTensor

    // these numbers are chosen to ensure consistency with 
    // the python implementation
    let deltas = tf.tensor(
        // batch size
        [
            // boxes
            [
                // coords
                [.1, .1, .1, .1],
                [-.2, -.2, .2, .2]
            ]
        ]
    ) as hacks.EasyTensor

    let outcome = tf.tensor(
        [
            [
                [0.02, 0.02, 1.02, 1.02],
                [0.496, 0.496, 0.604, 0.604]
            ]
        ]
    )

    test('RegressBoxes Layer class', async () => {
        let rb = new RegressBoxes({ anchorShape: boxes.shape })
        let inputs = [boxes, deltas]
        let result = rb.apply(inputs) as tf.Tensor
        assertTensorsEqual(result, outcome)
    })

    test('apply_bbox_deltas()', () => {
        let result = apply_bbox_deltas(boxes, deltas)
        assertTensorsEqual(result, outcome)
    })

    test('RegressBoxes Layer ClassName', () => {
        let rb = new RegressBoxes({ anchorShape: boxes.shape })
        expect(rb.getClassName()).toBe("RegressBoxes")
    });

    test('RegressBoxes Layer getComputedShape', () => {
        let rb = new RegressBoxes({ anchorShape: boxes.shape })
        let inputShape = [10, 10]

        let shape = rb.computeOutputShape()

        expect(shape).toStrictEqual(boxes.shape)
    });

})
