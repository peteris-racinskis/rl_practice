# Things done so far

Installed ROS, OpenAI gym

## In OpenAI gym

1. Got the example models and copy pasted RL code to work
2. Started from scratch and wrote PID controller for the Cartpole example
3. Wrote my own implementation of Q learning on top of the gym and cartpole example model (wrote my own discretization, update, policy, etc)

## Random notes
- Found out that changing directory name ruins the virtual environment - the path in the activation scripts remains unchanged. Needed to go in and change it to the current directory name to get the scripts to work correctly and install deps to the correct version of Python.
- Tensor shapes in tensorflow need to specify at least 2 dimensions, even if they're just vectors.