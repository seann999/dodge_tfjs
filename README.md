# Falling Balls + DQN
A demo of a DQN agent that learns to dodge falling balls, inspired by an old iPhone game.

The game utilizes [Matter.js](http://brm.io/matter-js/) while the neural network is built with [TensorFlow.js](https://js.tensorflow.org/).

Every frame, the agent senses the environment through raycasting. Actions are LEFT, STAY, and RIGHT. It receives a reward of +1 for living and -1 for dying.

On the webpage, hyperparameters can be edited at the bottom.

Note: Currently, resetting and restarting the agent seems to cause a small memory leak. Refreshing the page is recommended if this causes any problems.

[Webpage](http://web.sfc.keio.ac.jp/~t15704yn/falling/index.html)
