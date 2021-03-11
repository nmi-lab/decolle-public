#!/usr/bin/env bash

#Usage: grid_search.sh save_dir mem_required_per_experiment
#Kill all child processes on exit or ctrl-c
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

#Check if and how many GPUs are available
if hash nvidia-smi 2>/dev/null; then
    NUM_GPUS=`nvidia-smi --query-gpu=count --format="csv,noheader" | head -n 1`
else
    NUM_GPUS=0
fi
echo "Found ${NUM_GPUS} GPUs."

NUM_PROCS=0
MIN_REQ_MEM=${2-5000}
for filename in params_to_test/*; do
    if [ "${NUM_GPUS}" -gt "0" ] ;then
        
        # Look for available GPU
        while : ; do
            MAX_FREE_MEM=0
            
             for (( j=0; j<$NUM_GPUS; j++ )); do
                echo "Checking availability on GPU $j"
                MEM=0
                for i in {1..10}; do
                    ((MEM += `nvidia-smi --query-gpu=memory.free --format="csv,noheader,nounits" --id=${j}`))
                    sleep 1
                done
                ((MEM /= 10))
                if [ "$MEM" -gt "$MAX_FREE_MEM" ]; then
                    max_j=$j
                    MAX_FREE_MEM=$MEM
                fi
            done

            echo "The GPU with most free memory is GPU $max_j with $MAX_FREE_MEM MB"
            if [ "$MAX_FREE_MEM" -gt "$MIN_REQ_MEM" ]; then # If more than 10GB are free the n use that GPU
                break
            else
                echo "You need more than $MIN_REQ_MEM MB to launch the simulation."
            fi

            echo "All GPUS are busy. Waiting..."
            wait -n
        done

        # Running one simulation
        echo "Running on GPU number ${max_j}"
        [ ! -d "/path/to/dir" ] && mkdir -p logs/$1
        CUDA_VISIBLE_DEVICES=${max_j} python -u multilayer.py --params_file ${filename} --save_dir $1 >& logs/$1/`date +%d-%m-%Y_%H-%M-%S.log` &
	sleep 10

    else
        echo "Running on CPU"
        echo "python multilayer.py --params_file ${filename}"
    fi
done

echo "No more simulations to launch. Waiting for the ones currently running to finish"
wait # For the last simulations to finish
echo "All processes have finished their job. Exiting"

exit 0
