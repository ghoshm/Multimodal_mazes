#!/bin/bash

for i in {1..50}
do
    qsub HPC_array_job.pbs
done