'use strict';

var params;

var paramString = `params = {
  minibatchSize: 32,
  replayMemorySize: 10000,
  stackFrames: 2,
  targetUpdateFreq: 100,
  discount: 0.99,
  actionRepeat: 4,
  learningRate: 0.001,
  initExp: 1.0,
  finExp: 0.1,
  finExpFrame: 10000,
  replayStartSize: 100,

  numSensors: 20,
  sensorRange: 300,
  sensorDepthResolution: 3,
  
  hiddenLayers: [64, 64],
  activation: 'elu'
}`;

document.getElementById('settings').value = paramString;

var trainer = null;
var speedSlider = document.getElementById('speed');
var info = null;

var training = false;
var started = false;
var reset = false;

var model = null;
var targetModel = null;

var modelVars = null;
var replay = null;
var optimizer = null;
const maxEpisodeLength = 60*30;

function resetSignal() {
  reset = true;
}

function resetTrain() {
  reset = false;
  started = false;
  training = false;

  resetGame();

  data1 = new google.visualization.DataTable();
  data1.addColumn('number', 'X');
  data1.addColumn('number', 'Score');

  console.log("resetting; numTensors: " + tf.memory().numTensors);

  for (let i = 0; i < model.weights.length; i++) {
    model.weights[i].val.dispose();
  }

  for (let i = 0; i < targetModel.weights.length; i++) {
    targetModel.weights[i].val.dispose();
  }

  for(var key in optimizer) {
    if (optimizer[key]['isDisposed'] !== undefined) {
      optimizer[key].dispose();
    }
  }

  console.log(model);
  console.log(targetModel);

  console.log("reset; numTensors: " + tf.memory().numTensors);
  document.getElementById('start').innerHTML = 'Start';
}

function initTrain() {
  paramString = document.getElementById('settings').value;

  try {
    eval(paramString);
  } catch (err) {
    alert("Problem occured parsing parameters:\n" + err.message);
  }
  
  modelVars = [];
  replay = [];
  optimizer = tf.train.adam(params.learningRate);

  trainer = trainGen();

  console.log("building model...");
  targetModel = createModel();
  model = createModel();
  targetUpdate();

  for (let i = 0; i < model.weights.length; i++) {
    modelVars.push(model.weights[i].val);
  }

  /*for (let i = 0; i < model.weights.length; i++) {
    console.log(model.weights[i].val.dataSync());
  }*/
}

function toggleTrain() {
  if (!training) {
    console.log("starting; numTensors: " + tf.memory().numTensors);

    if (!started) {
      initTrain();
      console.log("init; numTensors: " + tf.memory().numTensors);

      setTimeout(trainUpdate, 0);

      started = true;
    }   

    training = true;
    document.getElementById('start').innerHTML = 'Pause';
  } else {
    training = false;
    document.getElementById('start').innerHTML = 'Resume';
  }
}

function createModel(stack) {
  const model = tf.sequential();
  
  model.add(tf.layers.dense({
    units: params.hiddenLayers[0],
    activation: params.activation,
    inputDim: (params.numSensors+1)*params.stackFrames
  }));

  for (let i = 0; i < params.hiddenLayers.length-1; i++) {
    model.add(tf.layers.dense({
      units: params.hiddenLayers[i+1],
      activation: params.activation,
      inputDim: params.hiddenLayers[i]
    }));
  }

  model.add(tf.layers.dense({
    units: N_ACTIONS,
    activation: 'linear',
    inputDim: params.hiddenLayers[params.hiddenLayers.length-1]
  }));

  return model;
}

function trainUpdate() {
  if (reset) {
    resetTrain();
  } else if (training) {
    for (let i = 0; i < Math.max(1, speedSlider.value); i++) {
      info = trainer.next().value;
    }
  }

  setTimeout(trainUpdate, Math.max(0, -speedSlider.value));
}

function targetUpdate() {
  console.log("updating target model");

  for (let i = 0; i < model.weights.length; i++) {
    targetModel.weights[i].val.assign(model.weights[i].val);
  }
}

function mse(predictions, targets, mask) {
  const e = tf.mul(predictions.sub(targets.expandDims(1)).square(), mask.asType('float32')).mean();
  return e;
}

function calcTarget(batchR, batchNextS, batchDone) {
  return tf.tidy(() => {
    const maxQ = targetModel.predict(batchNextS).max(1);
    const targets = batchR.add(maxQ.mul(tf.scalar(params.discount)).mul(batchDone));
    return targets;
  });
}

function* trainGen(episodes = 10000000) {
  console.log("training...");
  const scores = [];
  var totalFrames = 0;

  for (let ep = 0; ep < episodes; ep++) {
    var history = [resetGame()];
    var epDone = false;
    var epFrames = 0;
    const startTime = new Date().getTime();

    function stackObs() {
      const arrays = [];

      for (let i = 0; i < params.stackFrames; i++) {
        arrays.push(history[Math.max(0, history.length-1-i)]);
      }

      return Array.prototype.concat.apply([], arrays);
    }

    while (!epDone) {
      var act = Math.floor(Math.random()*N_ACTIONS);
      const observation = stackObs();
      const obsTensor = tf.tensor2d([observation]);
      const vals = model.predict(obsTensor);
      obsTensor.dispose();

      const a = Math.min(1, totalFrames/params.finExpFrame);

      if (Math.random() > a*0.1+(1-a)*params.initExp) {
        const maxAct = vals.argMax(1);
        act = maxAct.dataSync();
        maxAct.dispose();
      }

      var result = null;

      for (let t = 0; t < params.actionRepeat; t++) {
        result = step(act);
      }

      const normVals = tf.softmax(vals);

      yield {
        episode: ep,
        score: epFrames,
        observation: observation,
        reward: result.reward,
        values: vals.dataSync(),
        normValues: normVals.dataSync(),
        action: act
      };

      vals.dispose();
      normVals.dispose();

      history.push(result.sensors);

      epDone = result.gameOver || epFrames > maxEpisodeLength;

      replay.push({prevS: observation,
        action: act, reward: result.reward, nextS: stackObs(), done: epDone});

      if (replay.length > params.replayMemorySize) {
        replay = replay.slice(replay.length - params.replayMemorySize);
      }

      if (replay.length >= params.replayStartSize) {
        const loss = learn();

        if (result.gameOver) {
          const lossc = loss.dataSync()[0];
          console.log("loss: " + lossc);
          //data2.addRows([[ep, lossc]]);
        }
        loss.dispose();
      }

      epFrames++;
      totalFrames++;

      if (totalFrames % params.targetUpdateFreq === 0) {
        targetUpdate();

        console.log("frame: " + totalFrames);
        console.log("replay buffer: " + replay.length);
        console.log("numTensors: " + tf.memory().numTensors);
      }
    }

    scores.push(epFrames);
    const fps = epFrames/((new Date().getTime() - startTime)/1000);
    console.log("ep " + ep + ": survived " + epFrames + "; frames/second: " + fps);

    data1.addRows([[ep, epFrames]]);
    //data3.addRows([[ep, sps]]);
    updateGraphs();
  }
}

function learn() {
  const arrayPrevS = [];
  const arrayA = [];
  const arrayR = [];
  const arrayNextS = [];
  const arrayDone = [];

  for (let i = 0; i < params.minibatchSize; i++) {
    const exp = replay[Math.floor(Math.random() * replay.length)];
    arrayPrevS.push(exp.prevS);
    arrayA.push(exp.action);
    arrayNextS.push(exp.nextS);
    arrayR.push(exp.reward);
    arrayDone.push(exp.done ? 0 : 1);
  }

  const batchPrevS = tf.tensor2d(arrayPrevS);
  const batchA = tf.tensor1d(arrayA, 'int32');
  const batchR = tf.tensor1d(arrayR);
  const batchNextS = tf.tensor2d(arrayNextS);
  const batchDone = tf.tensor1d(arrayDone);
  
  const predMask = tf.oneHot(batchA, N_ACTIONS);

  const targets = calcTarget(batchR, batchNextS, batchDone);

  const loss = optimizer.minimize(() => {
    const x = tf.variable(batchPrevS);
    const predictions = model.predict(x);
    const re = mse(predictions, targets, predMask);
    x.dispose();

    return re;
  }, true, modelVars);

  targets.dispose();

  batchPrevS.dispose();
  batchA.dispose();
  batchR.dispose();
  batchNextS.dispose();
  batchDone.dispose();

  predMask.dispose();

  return loss;
}