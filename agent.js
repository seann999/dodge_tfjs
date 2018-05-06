var init_buffer = 100;
var batch_size = 32;
var MAX_REPLAY_BUFFER = 10000;
var speedSlider = document.getElementById("speed");
var skipSlider = document.getElementById("skip");
var stackSlider = document.getElementById("stack");
var obsFrames = 1;
var training = false;

function createModel(stack) {
  //const model = tf.sequential();
  /*model.add(tf.layers.dense(
    {units: 4, activation: 'relu', kernelInitializer: 'VarianceScaling', inputDim: (N_SENSORS+1)*obsFrames}));
  model.add(tf.layers.dense(
    {units: N_ACTIONS, activation: 'linear', kernelInitializer: 'VarianceScaling', inputDim: 4}));*/
  /*model.add(tf.layers.dense(
      {units: N_ACTIONS, activation: 'linear', inputDim: N_SENSORS+2}));*/
  const input = tf.input({shape: [(N_SENSORS+1)*obsFrames]});
  const linearLayer =
      tf.layers.dense({units: N_ACTIONS, useBias: true});
  const output = linearLayer.apply(input);
  const model = tf.model({inputs: input, outputs: output});

  return model;
}

function toggletrain() {
  training = !training;

  document.getElementById('start').innerHTML = training ? 'Pause' : 'Resume';
}

function train() {
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

    window.requestAnimationFrame(renderWorld);
    //setTimeout(makeSteps, 10);
  }

  function makeSteps() {
    if (training) {
      for (let i = 0; i < speedSlider.value; i++) {
        info = trainer.next().value;
      }
    }

    setTimeout(makeSteps, 1);
  }

  setTimeout(makeSteps, 1);
  window.requestAnimationFrame(renderWorld);
}

train();

var model = null;
var oldModel = null;
var info = null;

function freezeModel() {
  for (let i = 0; i < model.weights.length; i++) {
    oldModel.weights[i].val.assign(model.weights[i].val);
  }
}

var replay = [];

const learningRate = 0.001;
const optimizer = tf.train.adam(learningRate);

function lossFunc(predictions, targets, mask) {
  const mse = (tf.mul(predictions.sub(targets.expandDims(1)).square(), mask.asType('float32'))).mean();
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
  const batch_a = tf.tensor1d(array_a, dtype='int32');
  const batch_next_s = tf.tensor2d(array_next_s);
  const batch_r = tf.tensor1d(array_r);
  const batch_done = tf.tensor1d(array_done);

  const max_q = oldModel.predict(batch_next_s).max(1);
  const targets = batch_r.add(max_q.mul(tf.scalar(0.99)).mul(batch_done));

  const predMask = tf.oneHot(batch_a, N_ACTIONS);

  const loss = optimizer.minimize(() => {
    const x = tf.variable(batch_prev_s);
    const predictions = model.predict(x);

    console.log(batch_prev_s.dataSync())
    console.log(predictions.dataSync());
    console.log(targets.dataSync());
    //console.log(predMask.dataSync());
    console.log('--');

    const re = lossFunc(predictions, targets, predMask);
    x.dispose();
    return re;
  }, true);

  return loss;
}

obsFrames = stackSlider.value;
oldModel = createModel();
model = createModel();
console.log(model.predict(tf.ones([1, (N_SENSORS+1)*obsFrames])).dataSync());
//freezeModel();
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
    var exp = 0, gr = 0;
    var startTime = new Date().getTime();

    function stackObs() {
      const arrays = [];

      for (let i = 0; i < obsFrames; i++) {
        arrays.push(history[Math.max(0, history.length-1-i)]);
      }

      return [].concat.apply([], arrays);
    }

    while (!done) {
      var act = Math.floor(Math.random()*3);
      const observation = stackObs();
      const vals = model.predict(tf.tensor2d([observation]));

      if (Math.random() > Math.max(0.1, 1-ep/1000)) {
        act = vals.argMax(1).dataSync();
        gr += 1;
      } else {
        exp += 1;
      }

      var result = null;

      for (let t = 0; t < skipSlider.value; t++) {
        result = step(act);
      }

      const normVals = tf.softmax(vals);
      yield {observation: observation, reward: result.reward, values: normVals.dataSync(), action: act};
      vals.dispose();
      normVals.dispose();

      history.push(result.sensors);

      replay.push({prev_s: observation,
        action: act, reward: result.reward, next_s: stackObs(), done: result.gameOver});

      if (replay.length >= init_buffer) {
        const loss = tf.tidy(learn);
        if (result.gameOver) loss.print();
        loss.dispose();
      }

      if (replay.length > MAX_REPLAY_BUFFER) {
        replay = replay.slice(replay.length - MAX_REPLAY_BUFFER);
      }

      done = result.gameOver || frames > 60*30;
      frames++;
      steps++;

      if (step % 100 == 0) {
        freezeModel();
      }
    }

    scores.push(frames);
    const sps = frames/((new Date().getTime() - startTime)/1000);
    console.log("ep " + ep + ": survived " + frames + " steps/second: " + sps);
    //console.log(balls.length + " " + particles.length);
    data.addRows([[ep, frames]]);
    chart.draw(data);
    //console.log(scores);
  }
}

var data, chart;
google.charts.load('current', {packages: ['corechart', 'line']});
google.charts.setOnLoadCallback(drawBasic);

function drawBasic() {
  data = new google.visualization.DataTable();
  data.addColumn('number', 'X');
  data.addColumn('number', 'Score');

  var options = {
    hAxis: {
      title: 'Time'
    },
    vAxis: {
      title: 'Score'
    }
  };

  chart = new google.visualization.LineChart(document.getElementById('chart_div'));
  chart.draw(data, options);
}
