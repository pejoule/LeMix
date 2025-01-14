NUM_SAMPLES=334
BATCH_SIZE=1
MEMORY_THRESHOLD=0.7
SETTING=active  # "active" "isolated"
WORKLOAD=poisson
BASE_DIR=prof_main
ALPHA=0.15
BETA=0.05
EPSILON=0.2
K=0.5
EXPERIMENTS=1  # each experiment run 1 times
TOKEN="YOUR_ACCESS_TOKEN"
# Ensure the folder for profiling output exists
PROFILING_DIR="nvidia-profiling"
mkdir -p ${PROFILING_DIR}
NUM_NODES=4

for MODEL_NAME in "Llama-2-7b-chat-hf" "Llama-2-13b-chat-hf" "Llama-2-70b-chat-hf"; do
    for RATE_LAMBDA in 32 28 24 20 16 12 8 4; do
        OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/${MODEL_NAME}
        for RETRAIN_RATE in 0.1 0.3 0.5 0.7 0.9; do
            for TASK_ASSIGNMENT in "rr" "util" "workload"; do
                nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw --format=csv,nounits -l 1 -f ${PROFILING_DIR}/${TASK_ASSIGNMENT}_${MODEL_NAME}_${NUM_NODES}-node_${RATE_LAMBDA}-rps_${RETRAIN_RATE}-rate.csv &
                NVIDIA_SMI_PID=$!
                python distributed_llama.py \
                    --model_name_or_path "meta-llama/${MODEL_NAME}" \
                    --access_token $TOKEN \
                    --model_name $MODEL_NAME \
                    --num_nodes $NUM_NODES \
                    --n_samples $NUM_SAMPLES \
                    --rate_lambda $RATE_LAMBDA \
                    --workload $WORKLOAD \
                    --setting $SETTING \
                    --retraining_rate $RETRAIN_RATE \
                    --task_assignment $TASK_ASSIGNMENT \
                    --output_dir $OUTPUT_DIR \
                    --batch_size $BATCH_SIZE \
                    --alpha $ALPHA \
                    --beta $BETA \
                    --epsilon $EPSILON \
                    --k $K \
                    --experiments $EXPERIMENTS \
                    --memory_threshold $MEMORY_THRESHOLD \
                    --run_mode 'online'
                
                kill $NVIDIA_SMI_PID

                python plot.py \
                    --node $NUM_NODES \
                    --model_name $MODEL_NAME \
                    --setting $SETTING \
                    --workload $WORKLOAD \
                    --task_assignment $TASK_ASSIGNMENT \
                    --retraining_rate $RETRAIN_RATE \
                    --alpha $ALPHA \
                    --beta $BETA \
                    --epsilon $EPSILON \
                    --output_dir $OUTPUT_DIR
            done

            # ISOLATED
            nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw --format=csv,nounits -l 1 -f ${PROFILING_DIR}/separate_${MODEL_NAME}_${NUM_NODES}-node_${RATE_LAMBDA}-rps_${RETRAIN_RATE}-rate.csv &
            NVIDIA_SMI_PID=$!
            python distributed_llama.py \
                --model_name_or_path "meta-llama/$MODEL_NAME" \
                --access_token $TOKEN \
                --model_name $MODEL_NAME \
                --num_nodes $NUM_NODES \
                --n_samples $NUM_SAMPLES \
                --rate_lambda $RATE_LAMBDA \
                --workload $WORKLOAD \
                --setting "isolated" \
                --retraining_rate $RETRAIN_RATE \
                --task_assignment 'rr' \
                --output_dir $OUTPUT_DIR \
                --batch_size $BATCH_SIZE \
                --experiments $EXPERIMENTS \
                --memory_threshold $MEMORY_THRESHOLD

            kill $NVIDIA_SMI_PID

            python plot.py \
                --node $NUM_NODES \
                --model_name $MODEL_NAME \
                --setting "isolated" \
                --workload $WORKLOAD \
                --task_assignment 'rr' \
                --retraining_rate $RETRAIN_RATE \
                --output_dir $OUTPUT_DIR

        done
    done
done



