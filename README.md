### What's this repo about?

This is project #1 of Udacity's deep reinforcement learning nanodegree. The goal here is to create an agent that navigates a space littered with bananas and (hopefully) learns to collect yellow bananas (+1 reward for each) and avoid blue bananas (-1 reward for each).

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Dependencies

1. [Python 3.6](https://www.python.org/). It has to be 3.6 (3.5 or 3.7 won't work, because of PyTorch and Unity ML-Agents Toolkit).

2. [NumPy](http://www.numpy.org/)

3. [Matplotlib](https://matplotlib.org/)

4. [pandas](https://pandas.pydata.org/)

5. [PyTorch](https://pytorch.org/)

6. [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents), which is an open-source plugin that enables games and simulations to serve as environments for training intelligent agents. Download their Banana app from one of the links below.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - MacOS: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

### Replication instructions

Install all dependencies, download `main.py` and run it (`python main.py`).
