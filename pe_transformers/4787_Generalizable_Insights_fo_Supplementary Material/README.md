# GDT

## Installation

For a manual install you can run the following command using pip:
```
pip install torch==2.5.1 torch_geometric==2.6.0 numpy==1.26.4 torcheval==0.0.7 ogb==1.3.6 loguru==0.7.2 pandas==2.2.2 scikit-learn==1.5.1 networkx==3.3 tqdm==4.66.5 xformers==0.0.30 rdkit==2025.3.2
```
All experiments were conducted using PyTorch 2.5.1, PyTorchGeometric 2.6.0, Cuda Version 12.8 and Python Version 3.10


## Run Experiments 

We provide a bash file to run common experiments. ```run.sh``` contains the bashfile necessary to run the experiments for different datasets and hyperparameters. Common hyperparameters can be set in the command line using the following structure:

```bash
run.sh [task] [model] [model_size] [pe] [pe_steps] [seed] [root] [lr] [path]
```


Similarly, ```inference.sh``` contains the bashfile necessary to run inference experiments. In contrast to the previous hyperparameters common selections can be set with the following structure:

```bash 
inference.sh [inf_task] [task] [shots] [checkpoint] [model_size] [pe] [pe_steps] [seed] [root] 
```

task: denotes the task to be executed eg. coco, in case of inference the data source
inf_task: denotes the inference task to be executed (extrapolation, pascal, cycles)
model: selects the model, default value: transformer 
model_size: selects the model size, PCQ for 16M, PCQ_XL 90M, XL 160M
pe: selects the PE e.g., rwse, lpe
pe_steps: number of PE steps (random walks or eigenvalues) e.g., 16
seed: seed number to be used e.g., 14
root: root directory for datasets e.g., /data
lr: learning rate e.g., 0.0001
path: path for the result values to be saved e.g., /results
shots: number of shots in the few shot setting e.g., 50
checkpoint: path to checkpoint 


The following tasks are available:
pcqm4mv2: PCQ
coco: COCO
ogb-code2: Code
algo_reas_edge: Bridges
algo_reas_mst: MST
algo_reas_flow: Flow
cycles: cycles few shot
pascal: Pascal few shot


