MODEL_NAME=facebook/opt-125m
python main_opt.py \
      --model $MODEL_NAME \
      --save out/${MODEL_NAME}/orginial/
for PERCENTAGE in 0.001 0.003 0.00075 0.0005 0.005 0.0075
  do
    python main_opt.py \
      --model $MODEL_NAME \
      --save out/${MODEL_NAME}/bfa-mantissa/${PERCENTAGE}/ \
      --bfa \
      --bfa_percentage $PERCENTAGE

  done