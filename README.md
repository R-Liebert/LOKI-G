# LOKI-G

## Description
LOKI-G is a machine learning project that is a general adaptation of the [Locally Optimal search after K-step Imitation (LOKI)](https://arxiv.org/abs/1805.10413) algorithm. This project leverages both imitation learning (IL) and reinforcement learning (RL) to train a model. The model starts by learning from demonstration data through IL, then switches to RL for further training. The switching point can either be a fixed iteration or randomly selected based on a user-defined parameter.

LOKI-G uses [Closed-form Continuous-Time Neural Networks](https://arxiv.org/abs/2106.13898), and [Neural Circuit Policies](https://arxiv.org/abs/1803.08554). These resources are available in the ncps package.

## Installation
This project requires Python 3.7 or later. The dependencies can be installed with:

```bash
pip install -r requirements.txt
```

## Usage
First, adjust the parameters as needed in the `run_LOKI-G.sh` file. Then, you can start the training process with:

```bash
bash run_LOKI-G.sh
```
By default, the script is configured to run with a simulated environment, so it can be run without any special hardware.

## Arguments
- `--env_file`: Path to the Python file that defines the environment. By default, this is set to `src/env_files/simulated_env.py`, which runs a simple simulated environment. If you have a physical robot, you can switch to the robot environment by changing this path to `src/env_files/robot_env.py`. You may need to modify the `robot_env.py` file to match your hardware configuration.
- `--demonstration_path`: Directory containing the demonstration data (default: "./data").
- `--num_outputs`: Number of actions the model can output (default: 6).
- `--hard_switch_iter`: Iteration to switch from IL to RL during training (default: 18).
- `--random_sample_switch_iter`: If set, randomly choose the iteration to switch from IL to RL, with a distribution parameterized by `hard_switch_iter` (default: True).
- `--il_epochs`: Number of epochs to train the imitation learning model (default: 10).
- `--rl_epochs`: Number of epochs to train the reinforcement learning model (default: 10).
- `--render`: If set, render the environment during training.

## Results
The trained model is saved in the `saved_models` directory located one level above the directory where the script is run. The model is saved in TensorFlow format.

## License
This project is licensed under the terms of the MIT License.

## Contact
For any queries, please contact Robin Liebert [robinliebert@onmail.com] or Yuriy Yurchenko [yurchenkoyuriy97@gmail.com].
