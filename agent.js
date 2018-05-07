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
  var t = 0;
  var info = null;
  const trainer = traingen();
  const ctx = render.canvas.getContext("2d");

  function renderWorld() {
    if (info != null) {
      ctx.font = "30px Verdana";
      ctx.fillStyle = "white";
      ctx.fillText(info.episode, render.canvas.width - ctx.measureText(info.episode).width - 20, 30);

      ctx.font = "40px Verdana";
      ctx.fillStyle = "white";
      ctx.fillText(info.score, render.canvas.width - ctx.measureText(info.score).width - 20, 70);

      ctx.beginPath();
      var h = info.values[info.action]*10;

      if (h >= 0) {
        ctx.fillStyle = "DodgerBlue";
      } else {
        ctx.fillStyle = "Tomato";
      }

      ctx.rect(60, 80, h/10, 20);
      ctx.fill();

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
      for (let i = 0; i < info.normValues.length; i++) {
        ctx.beginPath();
        var shade = Math.floor(info.normValues[i] * 255);
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

    window.requestAnimationFrame(renderWorld);
  }

  function trainUpdate() {
    if (training) {
      for (let i = 0; i < Math.max(1, speedSlider.value); i++) {
        //let t1 = new Date().getTime();
        info = trainer.next().value;
        //data3.addRows([[t, (new Date().getTime() - t1)/1000]]);
        //t++;
      }
    }

    setTimeout(trainUpdate, Math.max(0, -speedSlider.value));
  }

  window.requestAnimationFrame(renderWorld);
  setTimeout(trainUpdate, 0);
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


  const targets = tf.tidy(() => {
    const max_q = oldModel.predict(batch_next_s).max(1);
    const targets = batch_r.add(max_q.mul(tf.scalar(0.99)).mul(batch_done));
    return targets;
  });
  
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

  batch_prev_s.dispose();
  batch_a.dispose();
  batch_next_s.dispose();
  batch_r.dispose();
  batch_done.dispose();
  targets.dispose();
  predMask.dispose();

  //await tf.nextFrame();

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
    var history = [resetGame()];
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

      if (Math.random() > Math.max(0.1, 1-ep/100)) {
        const maxact = vals.argMax(1);
        act = maxact.dataSync();
        maxact.dispose();
      }

      var result = null;

      for (let t = 0; t < skipSlider.value; t++) {
        result = step(act);
      }

      const normVals = tf.softmax(vals);
      //
      yield {
        episode: ep,
        score: frames,
        observation: observation,
        reward: result.reward,
        values: vals.dataSync(),
        normValues: normVals.dataSync(),
        action: act
      };
      //chart3.data.labels.push(steps);
      //

      obstensor.dispose();
      vals.dispose();
      normVals.dispose();

      history.push(result.sensors);

      if (history.length > obsFrames) {
        history = history.slice(history.length - obsFrames);
      }

      done = result.gameOver || frames > 60*30;

      replay.push({prev_s: observation,
        action: act, reward: result.reward, next_s: stackObs(), done: done});

      if (replay.length >= init_buffer) {
        let t1 = new Date().getTime();
        const loss = learn();
        data3.addRows([[ep, (new Date().getTime() - t1)/1000]]);

        if (result.gameOver) {
          const lossc = loss.dataSync()[0];

          //data2.addRows([[ep, lossc]]);
          //chart2.draw(data2, chart2opt);

          /*chart2.data.labels.push(ep);
          chart2.data.datasets[0].data.push(lossc);
          chart2.update();*/

          if (ep % 1 == 0) {
            data2.addRows([[ep, lossc]]);
            //chart2.options.data[0].dataPoints.push({ x: ep, y: lossc});
          }
        }
        loss.dispose();

        async function run() {
          await tf.nextFrame();
        }

        run();
      }

      frames++;
      steps++;

      if (steps % 100 === 0) {
        freezeModel();
        console.log("syncing models");
        console.log("replay buffer: " + replay.length);
        console.log("numBytes: " + tf.memory().numBytes);

        if (steps % 10000 == 0) {
          //chart1.update();
          //chart2.update();
          //chart3.update();

          //chart1.draw(data1, chart1opt);
          //chart2.draw(data2, chart2opt);
          //chart3.draw(data3, chart3opt);

          //chart1.render();
          //chart2.render();
          //chart3.render();
        }
        
      }
    }

    if (replay.length > MAX_REPLAY_BUFFER) {
      replay = replay.slice(replay.length - MAX_REPLAY_BUFFER);
    }

    scores.push(frames);
    const sps = frames/((new Date().getTime() - startTime)/1000);
    console.log("ep " + ep + ": survived " + frames + "; steps/second: " + sps);

    /*chart1.data.labels.push(ep);
    chart1.data.datasets[0].data.push(frames);
    chart1.update();

    chart3.data.labels.push(ep);
    chart3.data.datasets[0].data.push(sps);
    chart3.update();*/

    if (ep % 1 == 0) {
      //chart1.options.data[0].dataPoints.push({ x: ep, y: frames});
      //chart3.options.data[0].dataPoints.push({ x: ep, y: sps});
      data1.addRows([[ep, frames]]);
      //data3.addRows([[ep, sps]]);
    }
    
    //chart1.render();
    //chart3.render();
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

  chart1 = new google.visualization.LineChart(document.getElementById('scoreChart'));
  chart1.draw(data1, chart1opt);
}

function drawLosses() {
  data2 = new google.visualization.DataTable();
  data2.addColumn('number', 'X');
  data2.addColumn('number', 'Loss');

  chart2 = new google.visualization.LineChart(document.getElementById('lossChart'));
  chart2.draw(data2, chart2opt);
}

function drawTimes() {
  data3 = new google.visualization.DataTable();
  data3.addColumn('number', 'X');
  data3.addColumn('number', 'UPS');

  chart3 = new google.visualization.LineChart(document.getElementById('timeChart'));
  chart3.draw(data3, chart3opt);
}


function createChart(name, color, canvasid) {
  /*let canvas = document.getElementById(canvasid);
  //canvas.height = '100px';
  let ctx = canvas.getContext('2d');

  return new Chart(ctx, {
      type: 'line',

      data: {
          labels: [],
          datasets: [{
              label: name,
              //backgroundColor: 'rgb(255, 99, 132)',
              borderColor: color,
              data: [],
              lineTension: 0,
              fill: false
          }]
      },

      options: {
        //responsive: true,
        //maintainAspectRatio: false,
        scales: {
          xAxes: [{
            ticks: {
              autoSkip: true,
              maxTicksLimit: 20
            }
          }]
        },
        animation: false,
        elements: { point: { radius: 0 } }
      }
  });*/

  var chart = new CanvasJS.Chart(canvasid, {
    title:{
    text: name
    },
     data: [
    {
      lineColor: color,
      markerType: "none",
      type: "line",

      dataPoints: [
      ]
    }
    ]
  });

  chart.render();

  return chart;
}

//var chart1 = createChart('score', 'blue', 'scoreChart');
//var chart2 = createChart('loss', 'red', 'lossChart');
//var chart3 = createChart('time', 'orange', 'timeChart');

function updateGraphs() {
  //chart1.render();
  //chart2.render();
  //chart3.render();
  chart1.draw(data1, chart1opt);
  chart2.draw(data2, chart2opt);
  chart3.draw(data3, chart3opt);
}

