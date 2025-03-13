# AdaOff: An Adaptive Offloading Framework for Efficient LLM Inference on Local PCs

Dear readers, we have provided the source code to reproduce the results of our paper. The framework of AdaOff is developed based on the open-soured project llama.cpp. We construct the **Analyzer** in the file `analyzer.py`. To support our **Memory-First** technique, we build the dedicated CUDA operators in `cuda-op.cu `. Moreover, we also build the asynchronous communication kernel to implement the overlap in our **Order-Aware** technique.


We first list the dependencies of our code and then provide the commands to build this project and run the experiments mentioned in the Evaluation section of our paper.

# Dependency
`Ubuntu: 20.04`\
`CUDA: 12.4`\
`Python: 3.7.13`\
`Make: 4.2.1`\

# Build locally

make clean && LLAMA_CUBLAS=1 make -j

# Run
`./main -m path2model -n 500 -c 500 -p "Once upon" --n-gpu-layers 35 --ignore-eos -t 8 -s 1 --weight-ratio 1 --use-gpu`

`--weight-ratio`: control the volume of weights stored on GPU\
`--use-gpu`: if the offloaded ratio is lower than the turning point, our AdaOff would adopt the 
GPU-Centric inference manner then `--use-gpu` is true, vice versa.\
`--n-gpu-layers`: if total layer is n, when setting to n-1, AdaOff will offload the embedding weights,
when setting to n-2, AdaOff will offload the K cache, when setting to n-3, AdaOff will offload the K and V cache
`-n`: total context length
`-c`: the kv cache length
`-s`: fix the random seed
`-t`: the number of threads that CPU uses, only valid when AdaOff selects the CPU-Centric inference manner
