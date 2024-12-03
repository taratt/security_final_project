MODEL_NAME=facebook/opt-125m
python main_opt.py \
      --model $MODEL_NAME \
      --save out/${MODEL_NAME}/orginial/
for PERCENTAGE in 0.001 0.0001 0.00001 0.0005 0.005 0.01
  do
    python main_opt.py \
      --model $MODEL_NAME \
      --save out/${MODEL_NAME}/bfa-${PERCENTAGE}/ \
      --bfa \
      --bfa_percentage $PERCENTAGE

  done