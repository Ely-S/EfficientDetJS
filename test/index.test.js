const EfficientDet = require("../build/index.js").default
const tf = require("@tensorflow/tfjs")

test("Test EfficientDet Constructor ", () => {
    const model = new EfficientDet()
    expect(model).toBeInstanceOf(EfficientDet)
})

describe("EfficientDet", () => {
    const model = new EfficientDet()

    test("Test EfficientDet Load ", done => {
        model.load().then(done)
    })

    test("Test EfficientDet predict", async done => {
        await model.load()

        await model.predict(tf.ones([512, 512, 3]))

        done()
    })

})
