'use strict';

var init_buffer = 100;
var batch_size = 32;
var MAX_REPLAY_BUFFER = 10000;
var speedSlider = document.getElementById("speed");
var skipSlider = document.getElementById("skip");
var stackSlider = document.getElementById("stack");
var obsFrames = 1;
var training = false;

function createModel(stack) {
  const model = tf.sequential();
  model.add(tf.layers.dense(
    {units: 32, activation: 'relu', inputDim: (N_SENSORS+1)*obsFrames}));
  model.add(tf.layers.dense(
    {units: 32, activation: 'relu', inputDim: 32}));
  model.add(tf.layers.dense(
    {units: N_ACTIONS, activation: 'linear', inputDim: 32}));
  /*model.add(tf.layers.dense(
      {units: N_ACTIONS, activation: 'linear', inputDim: N_SENSORS+2}));*/

  return model;
}

function toggletrain() {
  training = !training;

  document.getElementById('start').innerHTML = training ? 'Pause' : 'Resume';
}

function train() {
  var info = null;
  var done = false;
  const trainer = traingen();
  const ctx = render.canvas.getContext("2d");

  function renderWorld() {
    if (info != null) {
      ctx.beginPath();
      ctx.fillStyle = "DodgerBlue";
      var h = info.values[info.action]*100;
      ctx.rect(30, 80, 200-h, 20);
      ctx.fill();

      /*ctx.beginPath();
      ctx.rect(400, 200-info.reward*100, 30, info.reward*100);
      ctx.fill();*/
      var offset = -10 + info.action * 10
      ctx.beginPath();
      ctx.fillStyle = "black";
      ctx.rect(player.position.x + offset - 15, player.position.y-10, 10, 10);
      ctx.rect(player.position.x + offset + 5, player.position.y-10, 10, 10);
      ctx.fill();

      /*for (var i = 0; i < N_SENSORS; i++) {
        var th = Math.PI - Math.PI / N_SENSORS * i;
        var mag = -(info.observation[i+2] * SENSOR_RESOLUTION - (SENSOR_RESOLUTION+1));
        ctx.beginPath();
        ctx.moveTo(player.position.x, player.position.y);
        ctx.lineTo(player.position.x + Math.cos(th)*mag*SENSOR_RANGE/SENSOR_RESOLUTION,
          player.position.y - Math.sin(th)*mag*SENSOR_RANGE/SENSOR_RESOLUTION);
        ctx.stroke();
      }*/

      var valSum = info.values.reduce(function(a, b) { return a + b; }, 0);
      for (let i = 0; i < info.values.length; i++) {
        ctx.beginPath();
        var shade = Math.floor(info.values[i] * 255);
        ctx.fillStyle = 'rgb(' + shade + ',' + shade + ',' + shade + ')';
        ctx.rect(30+i*30, 20, 20, 20);
        ctx.fill();
      }

      for (let i = 0; i < info.observation.length; i++) {
        ctx.beginPath();
        const col = Math.floor(i % (info.observation.length / obsFrames));
        const row = Math.floor(i / (info.observation.length / obsFrames));
        var shade = Math.floor(info.observation[i] * 255);
        ctx.fillStyle = 'rgb(' + shade + ',' + shade + ',' + shade + ')';
        ctx.rect(30+col*30, 50+row*(20/obsFrames), 20, 20/obsFrames);
        ctx.fill();
      }
    }

    if (training) {
      for (let i = 0; i < speedSlider.value; i++) {
        info = trainer.next().value;
      }
    }

    window.requestAnimationFrame(renderWorld);
  }

  window.requestAnimationFrame(renderWorld);
}

train();

var model = null;
var oldModel = null;

function freezeModel() {
  for (let i = 0; i < model.weights.length; i++) {
    oldModel.weights[i].val.assign(model.weights[i].val);
  }
}

var replay = [];

const learningRate = 0.001;
const optimizer = tf.train.adam(learningRate);

function lossFunc(predictions, targets, mask) {
  const mse = tf.mul(predictions.sub(targets.expandDims(1)).square(), mask.asType('float32')).mean();
  return mse;
}

function learn() {
  const array_prev_s = [];
  const array_a = [];
  const array_next_s = [];
  const array_r = [];
  const array_done = [];

  for (let i = 0; i < batch_size; i++) {
    const exp = replay[Math.floor(Math.random() * replay.length)];
    array_prev_s.push(exp.prev_s);
    array_a.push(exp.action);
    array_next_s.push(exp.next_s);
    array_r.push(exp.reward);
    array_done.push(exp.done ? 0 : 1);
  }

  const batch_prev_s = tf.tensor2d(array_prev_s);
  const batch_a = tf.tensor1d(array_a, 'int32');
  const batch_next_s = tf.tensor2d(array_next_s);
  const batch_r = tf.tensor1d(array_r);
  const batch_done = tf.tensor1d(array_done);

  const max_q = oldModel.predict(batch_next_s).max(1);
  const targets = batch_r.add(max_q.mul(tf.scalar(0.99)).mul(batch_done));

  const predMask = tf.oneHot(batch_a, N_ACTIONS);

  const loss = optimizer.minimize(() => {
    const x = tf.variable(batch_prev_s);
    const predictions = model.predict(x);

    //console.log(batch_prev_s.dataSync())
    //console.log(predictions.dataSync());
    //console.log(targets.dataSync());
    //console.log(predMask.dataSync());
    //console.log('--');

    const re = lossFunc(predictions, targets, predMask);
    x.dispose();
    return re;
  }, true);

  return loss;
}

obsFrames = stackSlider.value;
oldModel = createModel();
model = createModel();

freezeModel();
for (let i = 0; i < model.weights.length; i++) {
  console.log(model.weights[i].val.dataSync());
}

function* traingen(episodes = 100000) {
  console.log("building model...");

  console.log("training...");
  const scores = [];
  var steps = 0;

  for (let ep = 0; ep < episodes; ep++) {
    //console.log("episode: " + ep);
    const history = [resetGame()];
    var done = false;
    var frames = 0;
    const startTime = new Date().getTime();

    function stackObs() {
      const arrays = [];

      for (let i = 0; i < obsFrames; i++) {
        arrays.push(history[Math.max(0, history.length-1-i)]);
      }

      return Array.prototype.concat.apply([], arrays);
    }

    while (!done) {
      var act = Math.floor(Math.random()*N_ACTIONS);
      const observation = stackObs();
      const obstensor = tf.tensor2d([observation]);
      const vals = model.predict(obstensor);

      if (Math.random() > Math.max(0.1, 1-ep/1000)) {
        const maxact = vals.argMax(1);
        act = maxact.dataSync();
        maxact.dispose();
      }

      var result = null;

      for (let t = 0; t < skipSlider.value; t++) {
        result = step(act);
      }

      const normVals = tf.softmax(vals);
      const t1 = new Date().getTime();
      const r = result.reward;
      yield {observation: observation, reward: r, values: normVals.dataSync(), action: act};
      data3.addRows([[steps, (new Date().getTime()-t1)/1000.0]]);

      obstensor.dispose();
      vals.dispose();
      normVals.dispose();

      history.push(result.sensors);

      replay.push({prev_s: observation,
        action: act, reward: result.reward, next_s: stackObs(), done: result.gameOver});

      if (replay.length >= init_buffer) {
        tf.tidy(function() {
          const loss = tf.tidy(learn);
          if (result.gameOver) {
            const lossc = loss.dataSync()[0];
            data2.addRows([[ep, lossc]]);
            //chart2.draw(data2, chart2opt);
          }
          loss.dispose();
        });
      }

      if (replay.length > MAX_REPLAY_BUFFER) {
        replay = replay.slice(replay.length - MAX_REPLAY_BUFFER);
      }

      done = result.gameOver || frames > 60*30;
      frames++;
      steps++;

      if (steps % 100 === 0) {
        freezeModel();
        console.log("syncing models");
        console.log("replay buffer: " + replay.length);
        console.log("numTensors: " + tf.memory().numTensors);

        chart1.draw(data1, chart1opt);
        chart2.draw(data2, chart2opt);
        chart3.draw(data3, chart3opt);
      }
    }

    scores.push(frames);
    const sps = frames/((new Date().getTime() - startTime)/1000);
    console.log("ep " + ep + ": survived " + frames + "; steps/second: " + sps);
    //console.log(balls.length + " " + particles.length);
    data1.addRows([[ep, frames]]);
    //chart1.draw(data1, chart1opt);
    //data3.addRows([[ep, sps]]);
    //chart3.draw(data3, chart3opt);
    //console.log(scores);
  }
}

var data1, chart1, data2, chart2, data3, chart3;
google.charts.load('current', {packages: ['corechart', 'line']});
google.charts.setOnLoadCallback(drawScores);
google.charts.setOnLoadCallback(drawLosses);
google.charts.setOnLoadCallback(drawTimes);

var chart1opt = {
    hAxis: {
      title: 'Time'
    },
    vAxis: {
      title: 'Score'
    }
  };

var chart2opt = {
    hAxis: {
      title: 'Time'
    },
    vAxis: {
      title: 'Loss'
    },
    colors: ['red']
  };

var chart3opt = {
    hAxis: {
      title: 'Time'
    },
    vAxis: {
      title: 'Updates/Second'
    },
    colors: ['orange']
  };

function drawScores() {
  data1 = new google.visualization.DataTable();
  data1.addColumn('number', 'X');
  data1.addColumn('number', 'Score');

  chart1 = new google.visualization.LineChart(document.getElementById('score_div'));
  chart1.draw(data1, chart1opt);
}

function drawLosses() {
  data2 = new google.visualization.DataTable();
  data2.addColumn('number', 'X');
  data2.addColumn('number', 'Loss');

  chart2 = new google.visualization.LineChart(document.getElementById('loss_div'));
  chart2.draw(data2, chart2opt);
}

function drawTimes() {
  data3 = new google.visualization.DataTable();
  data3.addColumn('number', 'X');
  data3.addColumn('number', 'UPS');

  chart3 = new google.visualization.LineChart(document.getElementById('time_div'));
  chart3.draw(data3, chart3opt);
}
