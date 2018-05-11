# Falling Balls + DQN

![gif](https://raw.githubusercontent.com/seann999/dodge_tfjs/master/demo.gif)

A demo of a DQN agent that learns to dodge falling balls, inspired by an old iPhone game.

The game utilizes [Matter.js](http://brm.io/matter-js/) while the neural network is built with [TensorFlow.js](https://js.tensorflow.org/).

Every frame, the agent senses the environment through raycasting. Actions are LEFT, STAY, and RIGHT. It receives a reward of +1 for living and -1 for dying.

On the webpage, hyperparameters can be edited at the bottom.

Notes:
* Resetting and restarting the agent seems to cause a small memory leak. Refreshing the page is recommended if this causes any problems.
* Training will pause if the browser tab loses focus (depends on setTimeout).
* On default settings, the agent usually starts performing better at around 100 episodes.
* May not work correctly on some platforms/browsers. For example, for some reason, I have seen it not work with Chrome on Ubuntu 16.04, while it did with Firefox.

[Webpage](http://web.sfc.keio.ac.jp/~t15704yn/falling/index.html)
