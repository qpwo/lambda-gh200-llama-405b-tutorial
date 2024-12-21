

Lambda labs has half-off GH200s right now to get more people used to the ARM tooling. This means you can maybe actually afford to run the biggest open-source models! The only caveat is that you'll have to occasionally build something from source. Here's how I got llama 405b running with full precision on the GH200s.

## Create instances

Llama 405b is about 750GB so you want about 10 96GB GPUS to run it. (The GH200 has pretty good CPU-GPU memory swapping speed -- that's kind of the whole point of the GH200 -- so you can use as few as 3. Time-per-token will be terrible, but total throughput is acceptable, if you're doing batch-processing.) Sign in to lambda labs and create a bunch of GH200 instances. **Make sure to give them all the same shared network filesystem.**


![lambda labs screenshot](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/sjhtn33f1abnha6ng50c.png)

Save the ip addresses to ~/ips.txt.

## Bulk ssh connection helpers

I don't like using kubernetes or slurm; I prefer simple bash commands and ssh. We just need a couple short helpers.

```sh
all_ips=$(cat ~/ips.txt)
head_ip=$(head -n1 ~/ips.txt)
rest_ips=$(cat ~/ips.txt | tail -n+2)

# skip fingerprint confirmation
for ip in $all_ips; do
    echo "doing $ip"
    ssh-keyscan $ip >> ~/.ssh/known_hosts
done


run() {
    for ip in $ips; do
        ssh -i ~/.ssh/lambda_id_ed25519 ubuntu@$ip -- bash -l -c $(printf "%q" "$*") |& sed "s/^/$ip\t /" &
    done
    wait >/dev/null 2>&1
}
runall() { ips="$all_ips" run "$@"; }
runhead() { ips="$head_ip" run "$@"; }
runrest() { ips="$rest_ips" run "$@"; }

killall() {
    pkill -ife '.ssh/lambda_id_ed25519'
    sleep 1
    pkill -ife -9 '.ssh/lambda_id_ed25519'
    while [[ -n "$(jobs -p)" ]]; do fg || true; done
}

# Test all connections
runall echo runall works
runhead echo runhead works
runrest echo runrest works

# Output will be like
# [1] 256395
# [2] 256397
# ...
# 1.2.3.4   runall works
# 1.2.3.5   runall works
# 1.2.3.6   runall works
# ...

# check the GPUs are alright
runall nvidia-smi --query-gpu=power.draw,memory.used,memory.free --format=csv,noheader
```

### Set up NFS cache

We'll be putting the python environment and the model weights in the NFS. It will load much faster if we cache it.

```sh
# First, check the NFS works.
# runall ln -s my_other_fs_name shared
runhead 'echo world > shared/hello'
runall cat shared/hello

# Install and enable cachefilesd
runall sudo apt-get update
runall sudo apt-get install -y cachefilesd
runall "echo '
RUN=yes
CACHE_TAG=mycache
CACHE_BACKEND=Path=/var/cache/fscache
CACHEFS_RECLAIM=0
' | sudo tee -a /etc/default/cachefilesd"
runall sudo systemctl restart cachefilesd
runall 'sudo journalctl -u cachefilesd | tail -n2'

# Set the "fsc" option on the NFS mount
runhead cat /etc/fstab # should have mount to ~/shared
runall cp /etc/fstab etc-fstab-bak.txt
runall sudo sed -i 's/,proto=tcp,/,proto=tcp,fsc,/g' /etc/fstab
runall cat /etc/fstab

# Remount
runall sudo umount /home/ubuntu/wash2
runall sudo mount /home/ubuntu/wash2
runall cat /proc/fs/nfsfs/volumes # FSC column should say "yes"

# Test cache speedup
runhead dd if=/dev/urandom of=shared/bigfile bs=1M count=8192
runall dd if=shared/bigfile of=/dev/null bs=1M # First one takes 8 seconds
runall dd if=shared/bigfile of=/dev/null bs=1M # Seond takes 0.6 seconds
```

### Create conda environment

Instead of carefully doing the exact same commands on every machine, we can use a conda environment in the NFS and just control it with the head node.

```sh
# We'll also use a shared script instead of changing ~/.profile directly.
# Easier to fix mistakes that way.
runhead 'echo ". /opt/miniconda/etc/profile.d/conda.sh" >> shared/common.sh'
runall 'echo "source /home/ubuntu/shared/common.sh" >> ~/.profile'
runall which conda

# Create the environment
runhead 'conda create --prefix ~/shared/311 -y python=3.11'
runhead '~/shared/311/bin/python --version' # double-check that it is executable
runhead 'echo "conda activate ~/shared/311" >> shared/common.sh'
runall which python
```

### Install vllm dependencies

```sh
ssh ubuntu@$head_ip

pip install 'numpy<2' torch==2.5.1 --index-url 'https://download.pytorch.org/whl/cu124'
python -c 'import torch; print(torch.tensor(2).cuda() + 2, "torch ok")'

# fix for "libstdc++.so.6: version `GLIBCXX_3.4.30' not found" error
conda install -y -c conda-forge libstdcxx-ng=12

# if you trust my build you can just
pip install 'https://github.com/qpwo/triton-aarch64-wheel/releases/download/v3.2.0/triton-3.2.0+git755d4164-cp311-cp311-linux_aarch64.whl'
python -c 'import triton; print("triton ok")'

# otherwise, build it yourelf
function build_triton() {
    pip install -U pip setuptools wheel ninja cmake setuptools_scm
    git config --global feature.manyFiles true # faster clones
    git clone https://github.com/triton-lang/triton.git ~/shared/triton
    cd ~/shared/triton/python
    # git checkout 755d4164 # <-- version tested
    # Note that ninja already parallelizes everything to the extent possible,
    # so no sense trying to change the cmake flags or anything.
    python setup.py bdist_wheel
    pip install dist/*.whl # good idea to download this too for later
    python -c 'import triton; print("triton ok")'
}
```

### Install vllm

```sh
ssh ubuntu@$head_ip
git clone https://github.com/vllm-project/vllm.git ~/shared/vllm
cd ~/shared/vllm
python use_existing_torch.py
# git checkout c2d1b07 # <-- version tested
pip install -r requirements-build.txt
python use_existing_torch.py
python setup.py bdist_wheel
```


### Install aphrodite dependencies

I had the most luck building aarch64 packages by sticking to the lambda-provided system packages to the extent possible, and only adding a new package from apt or conda if necessary.

```sh
ssh ubuntu@$head_ip
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
conda install -y -c conda-forge libstdcxx-ng=12
conda install -y cmake sccache

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
python -c 'import torch; print(torch.tensor(5).cuda() + 1, "torch cuda ok")'

cd aphrodite-engine

pip install nvidia-ml-py==12.555.43 protobuf==3.20.2 ninja msgspec coloredlogs portalocker pytimeparse -r requirements-common.txt
pip install --no-clean --no-deps --no-build-isolation -v .

# if you want flash attention:
cd ..
git clone https://github.com/AlpinDale/flash-attention
cd flash-attention
pip install --no-clean --no-deps --no-build-isolation -v .
```

```sh
runhead pip install vllm
```
