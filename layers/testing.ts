import { Tensor } from "@tensorflow/tfjs"

export function assertTensorsEqual(t1: Tensor, t2: Tensor) {
    const ep = 1E-7
    let equals = t1.sub(t2)
        .less(ep)
        .all()
        .dataSync()

    expect(equals[0]).toBe(1)
}
