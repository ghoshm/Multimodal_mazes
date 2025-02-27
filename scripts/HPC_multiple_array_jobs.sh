#!/bin/bash

for i in {1..25}
do
    qsub HPC_array_job.pbs
done