nnodes=1                # number of nodes(=number of servers)
nproc_per_node=4        # number of processes per node(=number of GPUs per server)
MASTER_ADDR='localhost'
MASTER_PORT=29400
THREADS_PER_WORKER=12   # number of threads per process  

script=train_ddp.py


export OMP_NUM_THREADS=$THREADS_PER_WORKER && \
torchrun  --nnodes ${nnodes} \
          --nproc_per_node ${nproc_per_node} \
          --master_addr ${MASTER_ADDR} \
          --master_port ${MASTER_PORT} \
          $script


# Anyone looking into this in future, this is how I was able to solve it with nohup

# nohup sh run_ddp.sh > output.log 2>&1 &

# for example in this case
# nohup python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py ... > output.log 2>&1 &

# This should output you something like
# [1] 232423
# Here 1 is jobid and the big number is PID. Now you have to use the jobid with the disown command

# disown %1  
# This seems to be working for me so far