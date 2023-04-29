#!/bin/bash

unset PADDLE_TRAINER_ENDPOINTS
export CUDA_VISIBLE_DEVICES="4"

precision=${1:-"bf16"}
amp_level=${2:-"O2"}

MY_PYTHON_BIN=`which python`

cd $(dirname $0)
root_path="$(pwd)"

#export NCCL_DEBUG=INFO
export PYTHONPATH=$root_path:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/openmpi-3.1.0/lib:$LD_LIBRARY_PATH
TM_SCORE_BIN=$root_path/tools/tm_score
LDDT_SCORE_BIN=$root_path/tools/lddt

# disable C++ enisum, using python enisum
#export FLAGS_new_einsum=0
export FLAGS_use_autotune=1
export FLAGS_use_fast_math=1
#export FLAGS_check_nan_inf=1

chmod +x $TM_SCORE_BIN
chmod +x $LDDT_SCORE_BIN

#export GLOG_v=6
#export GLOG_vmodule=layer=4 #,dygraph_functions=5,utils=5
#export GLOG_vmodule=fused_gate_attention_op=4,fused_gate_attention=4,gpu_info=10,naive_best_fit_allocator=10,auto_growth_best_fit_allocator=10
#export GLOG_vmodule=fused_gate_attention_op=4,fused_gate_attention=4,layer=5
#export FLAGS_benchmark=1
#export CUDA_LAUNCH_BLOCKING=1
#export FLAGS_allocator_strategy="naive_best_fit"
#export FLAGS_fraction_of_gpu_memory_to_use=0.999

#export FLAGS_enable_gpu_memory_usage_log=1

train_af2() {
    start_step=5
    # distributed_args="-m paddle.distributed.launch --log_dir ./log/$exp_name"
    #profiler_type="native"
    #profiler_type="native-old"
    #profiler_type="nvprof"
    #profiler_type="debug"
    if [ "${profiler_type}" = "" ]; then
        profiler_type="none"
        train_step=105
    else
        train_step=35
    fi
    #train_step=3
    use_saved_train_batch="--use_saved_train_batch"
    if [ "${precision}" = "bf16" ]; then
      if [ "${amp_level}" = "O2" ]; then
        output_filename=helixfold.bs${batch_size}.${precision}_o2.${profiler_type}
      elif [ "${amp_level}" = "O1" ]; then
        output_filename=helixfold.bs${batch_size}.${precision}_o1.${profiler_type}
      fi
    else
      output_filename=helixfold.bs${batch_size}.${precision}.${profiler_type}
    fi
    if [ "${use_saved_train_batch}" != "" ]; then
        output_filename=${output_filename}.use_saved_data
    fi
    output_filename=${output_filename}.flashattn.dev20230424

    if [ "${profiler_type}" = "nvprof" ]; then
        export PATH=/opt/nvidia/nsight-systems/2022.5.1/bin:$PATH
        nsys_args="nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true -o ${output_filename}"
    fi

    #collect_gpu_status="True"
    if [ "${collect_gpu_status}" = "True" ]; then
        rm -rf gpu_usage_${output_filename}.txt
        nvidia-smi -i ${CUDA_VISIBLE_DEVICES} --query-gpu=utilization.gpu,utilization.memory --format=csv -lms 100 > gpu_usage_${output_filename}.txt 2>&1 &
        gpu_query_pid=$!
    fi

    echo "================================================="
    echo "data_config     : ${data_config}"
    echo "train_config    : ${train_config}"
    echo "batch_size      : ${batch_size}"
    echo "amp_level       : ${amp_level}"
    echo "precision       : ${precision}"
    echo "profiler_type   : ${profiler_type}"
    echo "output_filename : ${output_filename}"
    echo "================================================="

    ${nsys_args} ${MY_PYTHON_BIN} ${distributed_args} -u train.py \
            --tm_score_bin=${TM_SCORE_BIN} \
            --lddt_score_bin=${LDDT_SCORE_BIN} \
            --precision=${precision} \
            --amp_level=${amp_level} \
            --data_config=${data_config} \
            --train_config=${train_config} \
            --model_name=${model_name} \
            --init_model=${init_model} \
            --start_step=${start_step} \
            --train_step=${train_step} \
            --batch_size=${batch_size} \
            --dap_degree=${dap_degree} \
            --bp_degree=${bp_degree} \
            --num_workers 6 \
            --seed 2022 \
            --logging_level="DEBUG" \
            --model_dir="./models" \
            --log_dir="./outputs" \
            --profiler_type=${profiler_type} ${use_saved_train_batch} 2>&1 | tee log_${output_filename}.txt

    if [ "${collect_gpu_status}" = "True" ]; then
        kill ${gpu_query_pid}
    fi
}

exp_name="af2-init"
#exp_name="af2-finetune"

### initial_training
if [[ "$exp_name" == "af2-init" ]]; then
    train_config="./train_configs/initial.json"
    data_config="./data_configs/pdb-20211015.json"
    model_name="initial"
#    precision="bf16"
#    precision="fp32"
    batch_size=1
    dap_degree=1
    bp_degree=1
    train_af2
fi

### finetune training 
if [[ "$exp_name" == "af2-finetune" ]]; then
    train_config="./train_configs/finetune.json"
    data_config="./data_configs/pdb-20211015.json"
    model_name="finetune"
#    precision="bf16"
#    #precision="fp32"
    batch_size=1
    dap_degree=1
    bp_degree=1
    init_model="$root_path/data/af2_pd_params/model_5.pdparams"
    train_af2
fi
