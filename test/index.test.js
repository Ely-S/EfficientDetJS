const EfficientDet = require("../build/index.js").default

test("Test EfficientDet Constructor ", () => {
    const model = new EfficientDet()
    expect(model).toBeInstanceOf(EfficientDet)

})