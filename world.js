// load matter.js modules
var Engine = Matter.Engine,
    Render = Matter.Render,
    World = Matter.World,
    Body = Matter.Body,
    Bodies = Matter.Bodies,
    Common = Matter.Common,
    Composite = Matter.Composite,
    Events = Matter.Events;

// create engine
var engine = Engine.create();

// categories for collision masking
var defaultCategory = 0x0001,
        ballCategory = 0x0002,
        wallCategory = 0x0004,
        particleCategory = 0x0008,
        noCategory = 0x0016;

// keep array of just ball bodies for collision checking
var balls = [], ballBodies = [], particles = [];

const KEY_A = 65;
const KEY_D = 68;

const MAX_BALLS = 30;

var left = false, right = false;

var N_SENSORS = 20;
var N_ACTIONS = 2;//3;
// depth resolution
var SENSOR_RESOLUTION = 3;
// max depth
var SENSOR_RANGE = 300;

var myCanvas = document.getElementById('world');
var ctx = myCanvas.getContext("2d");

// create a renderer
var render = Render.create({
    canvas: myCanvas,
    //element: document.body,
    engine: engine,
    options: {
      wireframes: false,
      background: '#ccc'
    }
});

var gameOver = false;
var frame = 0;

function spawnBall() {
  var ball = Bodies.circle(Common.random(-400, -20),
    Common.random(-200, 0),
    Common.random(20, 60),
    {
      restitution: 1.0,
      friction: 0,
      collisionFilter: {
        category: ballCategory,
        mask: defaultCategory
      }
    });

  World.add(engine.world, ball);
  balls.push({body: ball, velX: Common.random(3, 10)});
  ballBodies.push(ball);
}

// init static bodies
var ground = Bodies.rectangle(-500, 600, (500+800)*2, 20, { isStatic: true });
var wallL = Bodies.rectangle(0, 0, 20, 1500, { isStatic: true, collisionFilter: {
  category: wallCategory
} });
var wallR = Bodies.rectangle(800, 0, 20, 1500, { isStatic: true, collisionFilter: {
  category: wallCategory
} });
World.add(engine.world, [ground, wallR, wallL]);

var player = null;

function resetGame() {
  gameOver = false;
  frame = 0;

  if (player) {
    World.remove(engine.world, player);
  }

  // init player
  player = Bodies.rectangle(400, 400, 50, 50, { friction: 0 });
  World.add(engine.world, player);

  // remove all balls
  for (var i = balls.length-1; i >= 0; i--) {
    var ball = balls[i];
    World.remove(engine.world, ball.body);
    balls.splice(i, 1);
    ballBodies.splice(i, 1);
  }

  // return initial observation
  return getSensors();
}

// splatter animation
function kill() {
  var newParts = [];
  for (var i = 0; i < 100; i++) {
    var p = Bodies.circle(player.position.x,
      player.position.y,
      Common.random(5, 10),
      {
        collisionFilter: {
          category: particleCategory,
          mask: noCategory
        },
        render: {
          fillStyle: "#f00"
        }
      });

    var ang = Math.PI*2*Math.random();
    var mag = Math.random() * 5 + 5
    Body.setVelocity(p, {x: Math.cos(ang)*mag, y: Math.sin(ang)*mag});
    newParts.push(p);
    particles.push(p);
  }
  World.add(engine.world, newParts);
}

resetGame();

Render.run(render);

function lethal(body) {
  return ballBodies.indexOf(body) != -1 || body == wallL || body == wallR;
}

Events.on(engine, 'collisionStart', function(event) {
    var pairs = event.pairs;

    for (var i = 0; i < pairs.length; i++) {
        var pair = pairs[i];

        if ((pair.bodyA == player && lethal(pair.bodyB))
          || (pair.bodyB == player && lethal(pair.bodyA))) {
            kill();
            gameOver = true;
        }
    }
});

function getSensors() {
  var sensors = [];

  // player position
  sensors.push(player.position.x / 800);

  for (var i = 0; i < N_SENSORS; i++) {
    var th = Math.PI - Math.PI / N_SENSORS * i;
    var hit = 0;

    for (var k = 1; k <= SENSOR_RESOLUTION; k++) {
        var results = Matter.Query.ray(ballBodies, player.position,
        {x: player.position.x + Math.cos(th)*k*SENSOR_RANGE/SENSOR_RESOLUTION,
          y: player.position.y - Math.sin(th)*k*SENSOR_RANGE/SENSOR_RESOLUTION}).length;

        if (results > 0) {
          hit = ((SENSOR_RESOLUTION+1)-k)/SENSOR_RESOLUTION;
          break;
        }
    }

    sensors.push(hit);
  }

  return sensors;
}

/*function renderWorld() {
  ctx.clearRect(0, 0, 300, 300);

  ctx.rect(player.position.x, player.position.y, 30, 30);
  ctx.stroke();

  ballBodies.forEach(body => {
    ctx.rect(body.position.x, body.position.y, 30, 30);
    ctx.stroke();
  });

  window.requestAnimationFrame(renderWorld);
}*/

function step(act, steps=1) {
  var velX = 0;
  if (act == 0) velX -= 10;
  else if (act == 1) velX += 10;

  Body.setVelocity(player, {x: velX, y: player.velocity.y});

  if (balls.length > MAX_BALLS) {
    for (var i = 0; i < MAX_BALLS-balls.length; i++) {
      var ball = balls[i];

      World.remove(engine.world, ball.body);
      balls.splice(i, 1);
      ballBodies.splice(i, 1);
    }
  }

  for (var i = 0; i < balls.length; i++) {
    var ball = balls[i];
    Body.setVelocity(ball.body, {x: ball.velX, y: ball.body.velocity.y});
  }

  for (var i = particles.length-1; i >= 0; i--) {
    var p = particles[i];

    if (p.position.y > 700) {
      World.remove(engine.world, p);
      particles.splice(i, 1);
    }
  }

  Engine.update(engine, 1000 / 60);

  var sensors = getSensors();

  if (frame % 60 == 0) {
    spawnBall();
  }

  frame++;

  var reward = 1;
  if (gameOver) reward = -1;

  return {sensors: sensors, reward: reward, gameOver: gameOver};
}

function autoStep() {
  var act = Math.floor(Math.random()*N_ACTIONS);
  var result = step(act);
  if (result.gameOver) {
    resetGame()
  }
}

window.addEventListener('keyup', (e) => {
  var code = e.keyCode;
  if (code == KEY_A) left = false;
  if (code == KEY_D) right = false;
}, false);

window.addEventListener('keydown', (e) => {
  var code = e.keyCode;
  if (code == KEY_A) left = true;
  if (code == KEY_D) right = true;
  if (code == 82) resetGame();
}, false);
