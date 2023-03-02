# STSN : Slot Transformer Scoring Network
Official repository for the paper - "[Learning to reason over visual objects](https://openreview.net/forum?id=uR6x8Be7o_M)." 



## Requirements
* python 3.9.7
* NVIDIA GPU with CUDA 11.0+ capability
* torch==1.11.0
* torchvision==0.12.0
* glob
* PIL==8.4.0
* numpy==1.20.3
* einops==0.4.1

## I-RAVEN
You can generate the dataset using the code from the [official repository](https://github.com/husheng12345/SRAN) or download it from [here](https://drive.google.com/file/d/1SxhImd29PLtlvqXAhlkH-CVDfFRzcK7y/view?usp=share_link).




To train on this dataset on a single 40GB A100 GPU, run the folowing command - `$ CUDA_VISIBLE_DEVICES=0 python train_slot_transformer_raven.py --batch_size 16 --img_size 80 --num_epochs 500 --run <RUN-NO-STR> --data <PATH-TO-DATASET> --save_path <PATH-TO-SAVE-MODEL>` 

## PGM

Download the data from the [official repository](https://github.com/deepmind/abstract-reasoning-matrices).

To train on this dataset, we used a cluster with multiple nodes (2) and multiple gpus per node (3), and submitting scripts (`pgm_job.slurm`) to the SLURM job scheduler.

You might have to change lines 4-8 of `pgm_job.slurm` depending on your cluster specifications. Default settings are for training on six 40GB A100 GPUs across 2 nodes.

Specify the path to the dataset, path to save the model, and run number (integer in string format) in the following line of `pgm_job.slurm`

`srun python train_slot_transformer_pgm_multigpu.py --img_size=80 --batch_size=16 --depth=24 --learning_rate=0.00008  --num_epochs=160 --run=<RUN-NO> --data=<PATH-TO-DATASET> --save_path=<PATH-TO-SAVE-MODEL> `

To train on the neutral regime run `$ sbatch pgm_job.slurm`

To resume training from a saved model checkpoint, uncomment line 271 of `train_slot_transformer_pgm_multigpu.py`, and turn off learning rate warmup by replacing
```
i += 1

if i < opt.warmup_steps:
 learning_rate = opt.learning_rate * (i / opt.warmup_steps)
else:
 learning_rate = opt.learning_rate
```
with `learning_rate = opt.learning_rate`, and specify `--model_checkpoint=<PATH-TO-SAVED-MODEL-CHECKPOINT>` in `pgm_job.slurm`

## CLEVR-Matrices

Download the data from [here](https://dataspace.princeton.edu/handle/88435/dsp01fq977z011).

To train on this dataset, we used a cluster with multiple nodes (2) and multiple gpus per node (4), and submitting scripts (`clevr_job.slurm`) to the SLURM job scheduler.

You might have to change lines 4-9 of `clevr_job.slurm` depending on your cluster specifications. Default settings are for training on eight 80GB A100 GPUs across 2 nodes.

Specify the path to the dataset, path to save the model, and run number (integer in string format) in the following line of `clevr_job.slurm`

`srun python train_slot_transformer_clevr_multigpu.py --batch_size=8  --num_epochs=200 --run=<RUN-NO> --data=<PATH-TO-DATASET> --save_path=<PATH-TO-SAVE-MODEL> `

To train run `$ sbatch clevr_job.slurm`

## Citation

We thank you for showing interest in our work. If our work was beneficial for you, please consider citing us using:
```
@inproceedings{mondallearning,
  title={Learning to reason over visual objects},
  author={Mondal, Shanka Subhra and Webb, Taylor Whittington and Cohen, Jonathan},
  booktitle={International Conference on Learning Representations}
}
```
