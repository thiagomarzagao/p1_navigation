### Result

The best result I managed to obtain was solving the taks in 664 episodes. The plot below shows how the scores as the episodes elapsed.

![scores X episodes](https://github.com/thiagomarzagao/p1_navigation/blob/master/Figure_1.png)

### Model

To achieve that result I used an adapted version of the [Deep Q-Learning](https://www.nature.com/articles/nature14236) model. The DQN is a neural network that approximates the optimal action-value function *Q* of conventional reinforcement learning algorithms. This approximation is necessary when the number of possible states is large (here it's infinite, as each of the 37 state dimensions is continuous). Hence instead of updating *Q(s,a)* at every step, as we would do in a conventional RL algorithm, here we update *Q(s,a,theta_i)*, where *theta_i* are the neural network's weights at iteratio *i*.

The architecture of the original DQN model consists of three convolutional layers followed by two fully connected layers. But he original DQN model uses images as inputs, which is not the case here (our 37-dimensional state represents the agent's velocity, position, etc, instead of pixels). Hence I removed the convolutional layers. Also, here our action space has only four dimensions (forward, backward, left, right), so that's the dimensionality of our output layer (as opposed to the 18 combinations of joystick direction and button of the original DQN model, which was meant to learn how to play Atari games).

More importantly, here I have kept only hidden layer. As Jeff Heaton argues in his book [Introduction to Neural Networks for Java](https://dl.acm.org/citation.cfm?id=1502373), *Problems that require two hidden layers are rarely encountered. However, neural networks with two hidden layers can represent functions with any kind of shape. There is currently no theoretical reason to use neural networks with any more than two hidden layers. In fact, for many practical problems, there is no reason to use any more than one hidden layer.*

I set the number of neurons of the hidden layer at 24, following the rule of thumb that the size of a hidden layer should be somewhere between that of the input layer (37) and that of the output layer (4), and ideally a power of 2 in order to facilitate computation. I tried other values (16, 32, 64, 128) but found little effect on the result.

I tried changing the hyperparameters but found no improvement.
