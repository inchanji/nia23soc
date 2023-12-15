nnodes=1                # number of nodes(=number of servers)
nproc_per_node=1        # number of processes per node(=number of GPUs per server)
MASTER_ADDR='localhost'
MASTER_PORT=29400
THREADS_PER_WORKER=12   # number of threads per process  

script=eval.py


export OMP_NUM_THREADS=$THREADS_PER_WORKER && \
torchrun  --nnodes ${nnodes} \
          --nproc_per_node ${nproc_per_node} \
          --master_addr ${MASTER_ADDR} \
          --master_port ${MASTER_PORT} \
          $script
