MODEL_NAME=facebook/opt-125m

for PERCENTAGE in  0.001 0.0005
  do
      python main_opt.py \
      --model $MODEL_NAME \
      --save out/${MODEL_NAME}/bfa/first_third/${PERCENTAGE}/ \
      --bfa \
      --first-third \
      --bfa_percentage $PERCENTAGE

    python main_opt.py \
      --model $MODEL_NAME \
      --save out/${MODEL_NAME}/bfa/second_third/${PERCENTAGE}/ \
      --bfa \
      --second-third \
      --bfa_percentage $PERCENTAGE

    python main_opt.py \
      --model $MODEL_NAME \
      --save out/${MODEL_NAME}/bfa/third_third/${PERCENTAGE}/ \
      --bfa \
      --third-third \
      --bfa_percentage $PERCENTAGE

    python main_opt.py \
      --model $MODEL_NAME \
      --save out/${MODEL_NAME}/bfa/atten/${PERCENTAGE}/ \
      --bfa \
      --atten\
      --bfa_percentage $PERCENTAGE

    python main_opt.py \
      --model $MODEL_NAME \
      --save out/${MODEL_NAME}/bfa/atten_out/${PERCENTAGE}/ \
      --bfa \
      --atten_out \
      --bfa_percentage $PERCENTAGE

    python main_opt.py \
      --model $MODEL_NAME \
      --save out/${MODEL_NAME}/bfa/fc/${PERCENTAGE}/ \
      --bfa \
      --fc \
      --bfa_percentage $PERCENTAGE

  done