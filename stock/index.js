const assert = require('assert')
const brain = require('brain.js')
const csv = require('csvtojson')
const fs = require('fs')

const stock = 'JKSE'
const net = new brain.recurrent.LSTMTimeStep({
  inputSize: 1,
  // inputRange: 20,
  outputSize: 1,
  hiddenLayers: [8, 8],
  decayRate: 0.999,
})

const getJSON = async () => {
  const json = await csv().fromFile(`./${stock}.csv`)
  const limited = json.slice(Math.max(json.length - 10, 1))
  let data = []
  let ioData = []
  let clsData = []
  let avg = 10000
  for (let i = 0; i < limited.length; i++) {
    const { Open, High, Low, Close } = limited[i]
    if (Open !== 'null' && High !== 'null' && Low !== 'null' && Close !== 'null') {
      if (avg > parseFloat(Low)) {
        avg = parseFloat(Low)
      }
      data.push({ open: parseFloat(Open), high: parseFloat(High), low: parseFloat(Low), close: parseFloat(Close), })
      ioData.push({ input: [parseFloat(Open), parseFloat(High), parseFloat(Low)], output: [parseFloat(Close)] })
    }
  }
  return { data, ioData, avg, clsData }
}

const training = async () => {
  console.log('start training...')
  const { ioData, clsData } = await getJSON()
  net.train([clsData], {
    log: true,
    errorThresh: 0.09,
    learningRate: 0.02
  })
  fs.writeFileSync(`state-${stock}.json`, JSON.stringify(net.toJSON()), 'utf8')

  const prediction = net.run([clsData])
  console.log(prediction)
}

// const forecastTraining = async () => {
//   console.log('start training...')
//   const rawData = await getJSON()
//   net.train(rawData, {
//     log: true,
//     errorThresh: 0.1,
//     // learningRate: 0.01
//   })
//   fs.writeFileSync(`forecast-${stock}.json`, JSON.stringify(net.toJSON()), 'utf8')

//   const prediction = net.forecast([trainingData], 1).map(scaleUp)
//   console.log(prediction)
// }

const scaleDown = ({ open, high, low, close }) => {
  return { open: open / avg, high: high / avg, low: low / avg, close: close / avg, }
  // return { input: [open / avg, high / avg, low / avg], output: close / avg }
}

const scaleUp = ({ open, high, low, close }) => {
  return { open: open * avg, high: high * avg, low: low * avg, close: close * avg, }
}

// const predicting = async () => {
//   console.log('start predicting...')
//   const rawData = await getJSON()
//   const trainingData = rawData.map(scaleDown)
//   try {
//     const file = fs.readFileSync(`state-${stock}.json`)
//     const json = JSON.parse(file)
//     net.fromJSON(json)
//     const prediction = net.forecast([trainingData], 1).map(scaleUp)
//     console.log(rawData)
//     console.log(prediction)
//   } catch (error) {
//     console.log(error)
//   }
// }


training()
// predicting()

