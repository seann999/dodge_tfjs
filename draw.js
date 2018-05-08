'use strict';

function renderWorld() {
  if (info != null) {
    ctx.font = "30px Verdana";
    ctx.fillStyle = "white";
    ctx.fillText(info.episode, render.canvas.width - ctx.measureText(info.episode).width - 20, 30);

    ctx.font = "40px Verdana";
    ctx.fillStyle = "white";
    ctx.fillText(info.score, render.canvas.width - ctx.measureText(info.score).width - 20, 70);

    ctx.beginPath();
    const h = info.values[info.action]*10;

    if (h >= 0) {
      ctx.fillStyle = "DodgerBlue";
    } else {
      ctx.fillStyle = "Tomato";
    }

    ctx.rect(60, 80, h/10, 20);
    ctx.fill();

    const offset = -10 + info.action * 10
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

    const valSum = info.values.reduce(function(a, b) { return a + b; }, 0);
    for (let i = 0; i < info.normValues.length; i++) {
      ctx.beginPath();
      const shade = Math.floor(info.normValues[i] * 255);
      ctx.fillStyle = 'rgb(' + shade + ',' + shade + ',' + shade + ')';
      ctx.rect(30+i*30, 20, 20, 20);
      ctx.fill();
    }

    for (let i = 0; i < info.observation.length; i++) {
      ctx.beginPath();
      const col = Math.floor(i % (info.observation.length / params.stackFrames));
      const row = Math.floor(i / (info.observation.length / params.stackFrames));
      const shade = Math.floor(info.observation[i] * 255);
      ctx.fillStyle = 'rgb(' + shade + ',' + shade + ',' + shade + ')';
      ctx.rect(30+col*30, 50+row*(20/params.stackFrames), 20, 20/params.stackFrames);
      ctx.fill();
    }
  }

  window.requestAnimationFrame(renderWorld);
}

window.requestAnimationFrame(renderWorld);

var data1, chart1, data2, chart2, data3, chart3;
google.charts.load('current', {packages: ['corechart', 'line']});
google.charts.setOnLoadCallback(drawCharts);

var chart1opt = {
  hAxis: {
    title: 'Episode'
  },
  vAxis: {
    title: 'Score'
  }
};

var chart2opt = {
  hAxis: {
    title: 'Episode'
  },
  vAxis: {
    title: 'Loss'
  },
  colors: ['red']
};

var chart3opt = {
  hAxis: {
    title: 'Episode'
  },
  vAxis: {
    title: 'Updates/Second'
  },
  colors: ['orange']
};

function drawCharts() {
  data1 = new google.visualization.DataTable();
  data1.addColumn('number', 'X');
  data1.addColumn('number', 'Score');

  chart1 = new google.visualization.LineChart(document.getElementById('scoreChart'));
  chart1.draw(data1, chart1opt);

  /*data2 = new google.visualization.DataTable();
  data2.addColumn('number', 'X');
  data2.addColumn('number', 'Loss');

  chart2 = new google.visualization.LineChart(document.getElementById('lossChart'));
  chart2.draw(data2, chart2opt);

  data3 = new google.visualization.DataTable();
  data3.addColumn('number', 'X');
  data3.addColumn('number', 'UPS');

  chart3 = new google.visualization.LineChart(document.getElementById('timeChart'));
  chart3.draw(data3, chart3opt);*/
}

function updateGraphs() {
  chart1.draw(data1, chart1opt);
  //chart2.draw(data2, chart2opt);
  //chart3.draw(data3, chart3opt);
}