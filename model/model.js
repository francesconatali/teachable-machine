const STATUS = document.getElementById('status');
const BUTTONS = document.getElementById('buttons');
const VIDEO = document.getElementById('webcam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
// Width and height of the expected input image for MobileNet
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];

TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

let dataCollectorButtons = document.querySelectorAll('button.dataCollector');

for (let i = 0; i < dataCollectorButtons.length; i++) {
  dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
  // Populate the human readable names for classes.
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}

let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;

/*
 * Loads the MobileNet model and warms it up so it's ready for use
 * with no latency.
 */
async function loadMobileNetFeatureModel() {
  const URL = 
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';

  mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});
  // STATUS.innerText = 'MobileNet v3 loaded successfully!';

  // Warm up the model by passing zeros through it once.
  tf.tidy(function () {
    let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    // Show the answer' shape in console for inspection.
    console.log(answer.shape);
  });
}

loadMobileNetFeatureModel();

// Create the additional layers for the model, the ones that will be trained.
let model = tf.sequential();
model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: CLASS_NAMES.length, activation: 'softmax'}));
model.summary();

// Compile the model with the defined optimizer and specify a loss function to use.
model.compile({
  // Adam changes the learning rate over time which is useful.
  optimizer: 'adam',
  // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
  // Else categoricalCrossentropy is used if more than 2 classes.
  loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy': 'categoricalCrossentropy', 
  // As this is a classification problem we can record accuracy in the logs.
  metrics: ['accuracy']  
});

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function enableCam() {
  if (hasGetUserMedia()) {
    // getUsermedia parameters.
    const constraints = {
      video: true,
      width: 640, 
      height: 480 
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
      VIDEO.srcObject = stream;
      STATUS.innerText = '';
      VIDEO.addEventListener('loadeddata', function () {
        videoPlaying = true;
      });
    }).catch(function (error) { 
        // Permission to use the webcam was denied.
        STATUS.innerText = 'Access to webcam denied';
    });
  } else {
    console.warn('getUserMedia() is not supported by your browser');
  }
}

enableCam();

/*
 * Handle Data Gather for button mouseup/mousedown.
 */
function gatherDataForClass() {
  let classNumber = parseInt(this.getAttribute('data-1hot'));
  gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
  
  dataGatherLoop();

  // Remove class 'processing' from TRAIN_BUTTON
  TRAIN_BUTTON.classList.remove('processing');
  // Set predict to true.
  predict = true;
}

function dataGatherLoop() {
  if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
    let imageFeatures = tf.tidy(function () {
      // Get the image data from the video element (webcam) and convert it to a tensor.
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
      // Resize the image to the size the MobileNet model expects.
      let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true);
      // Normalize the image to the range of the MobileNet model
      let normalizedTensorFrame = resizedTensorFrame.div(255);
      // Reshape the image to the shape the MobileNet model expects
      // and pass it through the model to get the features.
      return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
    });

    // Add the features extracted by mobilenet and the class to the training data
    // with which we will train the model later.
    trainingDataInputs.push(imageFeatures);
    trainingDataOutputs.push(gatherDataState);

    // Based on which class we are gathering data for, 
    // intialize array index element if currently undefined.
    if (examplesCount[gatherDataState] === undefined) {
      examplesCount[gatherDataState] = 0;
    }

    examplesCount[gatherDataState]++;

    // Update STATUS with the number of examples gathered for each class.
    STATUS.innerText = '';
    for (let n = 0; n < CLASS_NAMES.length; n++) {
      STATUS.innerText += ' ' + CLASS_NAMES[n] + ' snapshots count: ' + (examplesCount[n] ? examplesCount[n] : '0');
      // If not the last element, add ' - '
      if (n !== CLASS_NAMES.length - 1) {
        STATUS.innerText += ' - ';
      }
    }
    // Keep gathering data.
    window.requestAnimationFrame(dataGatherLoop);
  }
}

async function trainAndPredict() {
  // Continue only if predict = true.
  if (!predict) {
    return;
  }
  predict = false;
  // Gray out all the buttons while training by adding class 'processing'.
  BUTTONS.classList.add('processing');
  // Show status
  STATUS.innerText = 'Training...';
  // Shuffle the data.
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
  // Convert the data to tensors.
  let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
  // Convert the data to one-hot encoding.
  let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
  // Convert the data to tensors.
  let inputsAsTensor = tf.stack(trainingDataInputs);
  // Train the model.
  let results = await model.fit(inputsAsTensor, oneHotOutputs, {shuffle: true, batchSize: 5, epochs: 10, callbacks: {onEpochEnd: logProgress} });
  // Remove the gray out class from the buttons.
  BUTTONS.classList.remove('processing');
  // Dispose the tensors to free up memory.
  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();
  // Update predict to true.
  predict = true;
  predictLoop();
}

function logProgress(epoch, logs) {
  console.log('Data for epoch ' + epoch, logs);
}

function predictLoop() {
  if (predict) {
    tf.tidy(function () {
      // Get the image data from the video element (webcam) and convert it to a tensor and normalize
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
      // Resize the image to the size the MobileNet model expects
      let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor,[MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true);
      // Reshape the image to the shape the MobileNet model expects
      // and pass it through the model to get the features.
      let imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
      // Pass the features through the model previously created to get the predictions.
      let prediction = model.predict(imageFeatures).squeeze();
      // Get the index of the class with highest probability.
      let highestIndex = prediction.argMax().arraySync();
      // Get the array of predictions.
      let predictionArray = prediction.arraySync();

      STATUS.innerText = 'Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
    });
    // Keep predicting on the next frame.
    window.requestAnimationFrame(predictLoop);
  }
}

/*
 * Purge data and start over. Note: this does not dispose of the loaded 
 * MobileNet model and custom head tensors as they will be reused 
 * with a new traiing.
 */
function reset() {
  predict = false;
  // Add class 'processing' to TRAIN_BUTTON
  TRAIN_BUTTON.classList.add('processing');

  examplesCount.splice(0);
  for (let i = 0; i < trainingDataInputs.length; i++) {
    trainingDataInputs[i].dispose();
  }
  trainingDataInputs.splice(0);
  trainingDataOutputs.splice(0);
  STATUS.innerText = 'Data purged';
  // For inspection.
  console.log('Tensors in memory: ' + tf.memory().numTensors);
}
