docker container create -it -v ~/datasets:/datasets -v ./:/code --gpus all --shm-size 8g --name gridsearch grey-cuda
