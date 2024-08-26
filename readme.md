# AdvWFL

This repository contains the source code and datasets for the paper 'Adversarial Examples against WiFi Fingerprint-based Localisation in the Physical World'.
Our main work can be divided into three parts: normal localisation effects of CNNs in the WFL domain, attacks, and defences, and we have conducted experiments on the BHD and TUT datasets, for which the repository will provide the data and source code.
The two dataset and their source code is located in the `/BHD` and `TUT` directory respectively.

## Arguments
There are three mission types: normal, attack, and defense.
* normal: train + test
  ```python
  adv_test = False
  adv_train = False
  aux_batch = False
  pre_cluster = False
  ```
* attack: 
  ```python
  adv_test = True
  adv_train = False
  aux_batch = False
  pre_cluster = False
  ```
* Defence is divided into AdvT and RMBN methods:
  * AdvT:
    ```python
    adv_test = True
    adv_train = True
    aux_batch = False
    pre_cluster = False
    ```
  * RMBN:
    ```python
    adv_test = True
    adv_train = True
    aux_batch = True
    pre_cluster = True
    ```

`generate_model_type` and `test_model_type ` are parameters for generating models and testing models in the attack and defence phases, including `['AlexNet', 'VGG', 'ResNet']`

## Environment and Start-up
The conda environment can be created by executing the following command in the current directory
```text
conda create --name <env> --file environment.yml
```
After setting the arguments according to the type of task to be performed, run the main function in each dataset dictionary
```text
python main.py
```

