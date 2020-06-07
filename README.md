# AI-Lunar-Lander-TF-v1
A Reinforcement Learning AI Agent that use Deep Q Network to play Lunar Lander

* Implementation: Tensorflow v1
* Algorithm: Deep Q-Network with a single Fully Connected Neural Network.
* The agent has to learn how to land a Lunar Lander to the moon surface safely, quickly and accurately.
* If the agent just lets the lander fall freely, it is dangerous and thus get a very negative reward from the environment.
* If the agent does not land quickly enough (after 20 seconds), it fails its objective and receive a negative reward from the environment.
* If the agent lands the lander safely but in wrong position, it is given either a small negative or small positive reward, depending on how far from the landing zone is the lander.
* If the AI lands the lander to the landing zone quickly and safely, it is successful and is award very positive reward.


* Since the state space is infinite, traditional Q-value table method does not work on this problem. As a result, we need to integrate Q-learning with Neural Network for value approximation. However, the action space remains discrete.


