CONFIGS=(
    "configs/medio_albertino_muon.json"
    "configs/medio_albertino.json"
)

mkdir -p logs

for CONFIG_PATH in "${CONFIGS[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG_PATH" .json)
    echo "Inizio $CONFIG_NAME"

    python train_model.py \
        --config "$CONFIG_PATH" \
        > logs/${CONFIG_NAME}.log 2>&1

    if [ $? -eq 0 ]; then
        echo "Finito $CONFIG_NAME"
    else
        echo "Fallito $CONFIG_NAME"
    fi

done
