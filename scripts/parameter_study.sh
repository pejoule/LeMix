NUM_SAMPLES=1000
BATCH_SIZE=3
MEMORY_THRESHOLD=0.8
SETTING=active # "active" "isolated"
WORKLOAD=poisson
TASK_ASSIGNMENT=workload # "workload" "random" "rr" "util" "random+" "rr+" "util+"
LENGTH_DIST=random # "ascending" "bursty" "random" "descending"
BASE_DIR=prof_parameter
MODEL_NAME="Llama-2-70b-chat-hf"
TOKEN="YOUR_ACCESS_TOKEN"
K=0.5
EXPERIMENTS=1 # each experiment run 1 times
NUM_NODES=4


for RATE_LAMBDA in 10 20 30; do
    for RETRAIN_RATE in 0.1 0.3 0.5 0.7 0.9; do
        for ALPHA in 0.01 0.05 0.1 0.15 0.2 0.5 1.0; do
            for BETA in in 0.01 0.05 0.1 0.15 0.2 0.5 1.0; do
                for EPSILON in in 0.01 0.05 0.1 0.15 0.2 0.5 1.0; do
                    OUTPUT_DIR=${BASE_DIR}/${NUM_NODES}_node/lambda_${RATE_LAMBDA}/$MODEL_NAME
                    python distributed_llama.py \
                        --model_name_or_path "meta-llama/$MODEL_NAME" \
                        --access_token $TOKEN \
                        --model_name $MODEL_NAME \
                        --num_nodes $NUM_NODES \
                        --n_samples $NUM_SAMPLES \
                        --rate_lambda $RATE_LAMBDA \
                        --workload $WORKLOAD \
                        --setting $SETTING \
                        --retraining_rate $RETRAIN_RATE \
                        --task_assignment "workload" \
                        --output_dir $OUTPUT_DIR \
                        --batch_size $BATCH_SIZE \
                        --alpha $ALPHA \
                        --beta $BETA \
                        --epsilon $EPSILON \
                        --k $K \
                        --experiments $EXPERIMENTS \
                        --memory_threshold $MEMORY_THRESHOLD

                    python plot.py \
                        --node $NUM_NODES \
                        --model_name $MODEL_NAME \
                        --setting $SETTING \
                        --workload $WORKLOAD \
                        --task_assignment "workload" \
                        --retraining_rate $RETRAIN_RATE \
                        --alpha $ALPHA \
                        --beta $BETA \
                        --epsilon $EPSILON \
                        --output_dir $OUTPUT_DIR
                done
            done
        done
    done
done
