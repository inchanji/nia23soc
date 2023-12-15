
path2data=/home/data/nia23soc
# docker run -ti docker-nia23soc:latest /bin/bash
docker run --memory=68g --memory-swap=68g --shm-size=16g -v $path2data:$path2data -ti docker-nia23soc /bin/bash

# export LANG=ko_KR.UTF-8