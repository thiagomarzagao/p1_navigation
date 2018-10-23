### Result

The best result I managed to obtain was solving the taks in 664 episodes. The plot below shows how the scores as the episodes elapsed.

![scores X episodes](https://github.com/thiagomarzagao/p1_navigation/blob/master/Figure_1.png)

### Model

To achieve that result I used an adapted version of the [Deep Q-Learning](https://www.nature.com/articles/nature14236) model. The DQN is a neural network that approximates the Q values of conventional reinforcement learning algorithms. This approximation is necessary when the number of possible states is large (here it's infinite, as each of the 37 state dimensions is continuous). Hence instead of updating *Q(s,a)* at every step, as we would do in a conventional RL algorithm, here we update *Q(s,a,theta)*
