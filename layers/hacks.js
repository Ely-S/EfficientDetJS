
module.exports.init = function init(TensorClass) {
  if (TensorClass.prototype.$) return TensorClass

  TensorClass.prototype.$ = function $(selection) {
    // translate from np to tfjs
    // e.g.
    // boxes[:, :, 1]
    // becomes
    // boxes.slice([0, 0, 2], [-1, -1, 1])

    if (!cache.hasOwnProperty(selection)) {
      cache[selection] = getBeginSizeAxes(selection)
    }

    var [begin, size, axes_to_squeeze] = cache[selection]

    let sliced = this.slice(begin, size)

    if (axes_to_squeeze.length) {
      let squeezed = sliced.squeeze(axes_to_squeeze)
      sliced.dispose()
      return squeezed
    }

    return sliced
  }
}

const cache = {}

function getBeginSizeAxes(selection) {
  var begin = []

  var size = []

  var axes_to_squeeze = []

  selection.replace(/\s/g, '').split(',').forEach((slice, index) => {
    if (/^[0-9]+$/.test(slice)) {
      let start = parseInt(slice, 10)
      begin.push(start)
      size.push(1)
      axes_to_squeeze.push(index)
    } else if (slice === ':') {
      begin.push(0)
      size.push(-1)
    } else if (slice.startsWith(':')) {
      // only support forward slices with no negative indices
      let end = slice.split(':')[1]
      begin.push(0)
      size.push(parseInt(end, 10))
    } else {
      let [startS, endS] = slice.split(':')
      let start = parseInt(startS, 10)
      let end = parseInt(endS, 10)
      let sliceSize = end - start

      begin.push(start)
      size.push(sliceSize)
    }
  })

  return [begin, size, axes_to_squeeze]

}
