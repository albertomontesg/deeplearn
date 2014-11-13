#!/bin/bash

# @ job_name = Encoder_10002
# @ initialdir = ./
# @ output = outputs/EncoderTraining_10002.out <- canviar per un directori
# @ error = errors/EncoderTraining_10002.err <- Canviar per un directori
# @ wall_clock_limit = 00:05:00
# @ cpus_per_task = 12
# @ tasks_per_node = 1
# @ total_tasks = 1
# @ gpus_per_node = 2
## @ partition = debug

module purge
module load oscar-modules/1.0.3 transfer/1.0 gcc/4.6.1 bullxmpi/bullxmpi-1.1.11.1 cuda/5.0 atlas/3.10.2 python/2.7.8

python SingleEncoderTraining.py 10002 3 0.050000 tanh 5 10
