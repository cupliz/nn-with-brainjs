const assert = require('assert')
const brain = require('brain.js')
const csv = require('csvtojson')
const fs = require('fs')
const moment = require('moment')

const stock = 'JKSE'
let min = 10000

const unixDate = (date) => moment(date, 'YYYY-MM-DD').unix()
const getJSON = async () => {
  const json = await csv().fromFile(`./data/${stock}.csv`)
  const limited = json.slice(Math.max(json.length - 100, 1))
  let data = []
  for (let i = 0; i < limited.length; i++) {
    const { Open, High, Low, Close, Date } = limited[i]
    if (Open !== 'null' && High !== 'null' && Low !== 'null' && Close !== 'null') {
      const date = parseFloat(unixDate(Date))
      const open = parseFloat(parseFloat(Open).toFixed(2))
      const high = parseFloat(parseFloat(High).toFixed(2))
      const low = parseFloat(parseFloat(Low).toFixed(2))
      const close = parseFloat(parseFloat(Close).toFixed(2))
      if (low < min) {
        min = low
      }
      data.push({ input: [open, high, low], output: [close] })
    }
  }
  return { data }
}

// const scaleDown = ({ open, high, low, close }) => {
//   return { open: open / min, high: high / min, low: low / min, close: close / min, }
// }

// const scaleUp = ({ open, high, low, close }) => {
//   return { open: open * min, high: high * min, low: low * min, close: close * min, }
// }

const config = {
  // binaryThresh: 0.5,
  // decayRate: 0.999,
  // inputRange: 20,
  inputSize: 3,
  outputSize: 1,
  hiddenLayers: [8, 8],
}
const tConfig = {
  log: true,
  errorThresh: 0.09,
  learningRate: 0.01
}
const net = new brain.recurrent.LSTM(config)

// 2019-12-23, 6309.670898, 6315.721191, 6270.539063, 6305.910156, 6305.910156, 39077900
const training = async () => {
  console.log('--- start training...')
  const { data } = await getJSON()

  net.train(data, tConfig)

  fs.writeFileSync(`./state/${stock}.json`, JSON.stringify(net.toJSON()), 'utf8')
}
// const emulate = () => {
//   console.log('--- predicting...')
//   const prediction = net.run([6309.67, 6315.72, 6270.54]) // 6305.91
//   console.log(prediction)
// }
const emulateFromCache = () => {
  console.log('--- predicting...')
  const file = fs.readFileSync(`./state/${stock}.json`)
  const json = JSON.parse(file)
  net.fromJSON(json)
  const prediction = net.run([6309.67, 6315.72, 6270.54]) // 6305.91
  console.log(prediction)
}
const init = async () => {
  await training()
  emulateFromCache()
}
init()

