# GIM-RL: Generic Itemset Mining based on Reinforcement Learning

**[Overview](#overview)** | **[Setup](#setup)** | **[Examples](#examples)**

## Overview
These source codes are for GIM-RL, an algorithm for itemset mining using a reinforcement learning agent.  
This algorithm supports the tasks of High Utility Itemset Mining, Frequent Itemset Mining, and Association Rule Mining, and can mine itemsets by determining the dataset to be mined and the threshold value for each task.
This algorithm offers a unified framework to extract various types of itemsets.
One can easily extend the source codes to extract a different type of itemsets by defining his/her own reward.

## Setup
It is recommended to build the source codes on an environment using docker which has the following dependencies.

- GPUs that support CUDA and nvidia-drivers
- docker ( >= 19.03)
- nvidia-docker
- make

Please follow the steps below to set up.

1. Build a docker image, create a container, and attach it to the container.

    ```bash
    make up
    ```

2. Download the dataset

    ```bash
    ./download_data.sh
    ```

3. Start MLflow using tmux or background execution

    ```bash
    mlflow ui --host 0.0.0.0
    ```

After starting MLflow, access `localhost:5000` with a browser and check the execution record from there.
Use the `exit` command to exit from the container, and the `make up` command to reattach to the container.



## Examples
Hydra is used to manage the parameter, and the results are aggregated in MLflow and can be checked from a browser.  
Hydra is used to select a dataset and agent to be used, and to configure various parlermeters.  
Detailed parameters are described in [config.yaml](config/config.yaml). In addition, by installing [Joblib Launcher plugin](https://hydra.cc/docs/plugins/joblib_launcher), parallel execution is possible using the `-m` parameter.

### High Utility Itemset Mining
Extract itemsets formed by highly profitable items from a dataset.  
You can use Chess, Mushroom, Accidents_10%, and Connect as datasets.

Run: hui_train.py
```bash
python hui_train.py dataset/hui=chess,mushroom,accidents10per,connect -m
```

### Frequent Itemset Mining
Extract itemsets consisting of frequently co-occurring items from a dataset.  
Chess, Mushroom, Pumsb, and Connect can be used as datasets. 

Run: fp_train.py
```bash
python fp_train.py dataset/fp=chess,mushroom,pumsb,connect agent.lambda_end=0.6 agent.network=simple interaction.episodes=1000 -m
```

### Association Rule Mining
Extract rules consisting of correlated items from a dataset.  
Chess, Mushroom, Pumsb, and Connect can be used as datasets.  

Run: ar_train.py
```bash
python ar_train.py dataset/ar=chess,mushroom,pumsb,connect interaction.episodes=1000 -m
```

### Agent Transfer
Apply an agent trained on a source dataset to another target dataset.
You can use the dataset available for each mining task.
The first 60% and the remaining 40% of the dataset are used as the source and target partitions, respectively.

Run: transfer_hui_train.py
```bash
python transfer_hui_train.py dataset/transfer=hui_chess,hui_mushroom,hui_accidents10per,hui_connect agent.test_lambda_start=0.5 -m
```

Run: transfer_fp_train.py
```bash
python transfer_fp_train.py dataset/transfer=fp_chess,fp_mushroom,fp_pumsb,fp_connect agent.lambda_end=0.6 agent.test_lambda_end=0.6 agent.network=simple -m
```

Run: transfer_ar_train.py
```bash
python transfer_ar_train.py dataset/transfer=ar_chess,ar_mushroom,ar_pumsb,ar_connect agent.lambda_start=0.5 -m
```

## Note
If you get an error when installing the libraries in the setup or cannot create the container, you can delete the pip entry in [dev_env.yml](docker/dev_env.yml) and run it again. The dependencies without using docker are the following

- Python 3.7
- [PyTorch](https://pytorch.org/) 1.5
- [Numpy](https://numpy.org/) (>1.18.1)
- [Gym](https://gym.openai.com/) 0.17.2
- [Hydra](https://hydra.cc/) 1.0.0
- [MLflow](https://mlflow.org/) 1.11.0
- requests
- yaml

It is also recommended to run these source codes on a GPU, but if a GPU is unavailable, it is also possible to not use a GPU by changing the base image in the [Dockerfile](docker/Dockerfile) to the one that does not use CUDA, and by removing the `--gpus all` parameter in the [Makefile](Makefile).

## Reference
If you found this code useful, please cite the following paper:

```
@article{Fujioka_GIM,
  author = {Fujioka, Kazuma and Shirahama, Kimiaki},
  title = {Generic Itemset Mining Based on Reinforcement Learning},
  journal = {arXiv e-prints, arXiv:2105.07753},
  year = {2021}
}
```

## License
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.html)
