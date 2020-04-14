#! /bin/sh

#where do you like to store the trained models?
export OUTPUT_GLOBAL_DIR=/projectnb/llamagrp/feyzanb
#where do you have the downloaded dataset?
export DATA_GLOBAL_DIR=/project/llamagrp/feyza/transformers/dataset
#where do you have the dataset for baselines i.e. 9 binary classifiers?
#no need to provide if you're not running the particular experiment.
export BASELINE_DATA_GLOBAL_DIR=/project/llamagrp/feyza/transformers/baselines/dataset
#where do you like to store the downloaded pretrained BERT models initially?
export CACHE_GLOBAL_DIR=/projectnb/llamagrp/feyzanb/.cache/newsframing


EXPERIMENT=$1
##################################################
#                 Table 1 
##################################################

case "$EXPERIMENT" in

##################################################
#          Table 1 Experiment 1
##################################################

    "Table1Exp1") 
    export EXP_NAME=multiclass_engbert_softmaxfocal
    echo "Running $EXP_NAME"

    for i in 0 1 2 3 4
    do
      python main.py --model_type=bertmultilabel \
                      --data_dir=$DATA_GLOBAL_DIR/dataset_multiclass_wide/$i \
                      --task_name=frame --model_name_or_path=bert-base-uncased \
                      --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i \
                      --cache_dir=$CACHE_GLOBAL_DIR/english \
                      --per_gpu_train_batch_size=4 \
                      --learning_rate=2e-5 \
                      --num_train_epochs=10.0 \
                      --max_seq_length=128 \
                      --loss=softmax-focal \
                      --exp_name=$EXP_NAME \
                      --do_train \
                      --do_eval \
                      --do_lower_case \
                      --overwrite_output_dir \
                      --set_weights
      echo "Finished fold $i"
    done

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                              --exp_name=$EXP_NAME
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=multiclass \
                           --data_path=$DATA_GLOBAL_DIR/dataset_multiclass_wide

    echo "Saved predictions to $EXP_NAME.pkl" ;;
    
##################################################
#          Table 1 Experiment 2
##################################################

    "Table1Exp2")
    
    export EXP_NAME=multiclass_multibert_softmaxfocal
    echo "Running $EXP_NAME"

    for i in 0 1 2 3 4
    do
      python main.py --model_type=bertmultilabel \
                     --data_dir=$DATA_GLOBAL_DIR/dataset_multiclass_wide/$i \
                     --task_name=frame --model_name_or_path=bert-base-multilingual-cased \
                     --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i \
                     --cache_dir=$CACHE_GLOBAL_DIR/multilingual \
                     --per_gpu_train_batch_size=4 \
                     --learning_rate=2e-5 \
                     --num_train_epochs=10.0 \
                     --max_seq_length=128 \
                     --loss=softmax-focal \
                     --exp_name=$EXP_NAME \
                     --do_train \
                     --do_eval \
                     --overwrite_output_dir \
                     --set_weights
      echo "Finished fold $i"
    done

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                              --exp_name=$EXP_NAME
                              
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=multiclass \
                           --data_path=$DATA_GLOBAL_DIR/dataset_multiclass_wide

    echo "Saved predictions to $EXP_NAME.pkl" ;;
    
##################################################
#          Table 1 Experiment 3
##################################################

    "Table1Exp3")
    # todo
    ;;
    
##################################################
#          Table 1 Experiment 4
##################################################

    "Table1Exp4")
    export EXP_NAME=binary_multibert_focal_9binary_classifiers
    echo "Running $EXP_NAME"

    for j in 1 2 3 4 5 6 7 8 9
    do
      for i in 0 1 2 3 4
      do
        python 9binaryclassifiers.py --model_type=bertmultilabel \
                                     --data_dir=$BASELINE_DATA_GLOBAL_DIR/frame$j/$i \
                                     --task_name=frame \
                                     --model_name_or_path=bert-base-multilingual-cased \
                                     --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/frame$j/$i \
                                     --cache_dir=$CACHE_GLOBAL_DIR/multilingual \
                                     --per_gpu_train_batch_size=4 \
                                     --learning_rate=2e-5 \
                                     --num_train_epochs=10.0 \
                                     --max_seq_length=128 \
                                     --do_eval \
                                     --do_train \
                                     --overwrite_output_dir \
                                     --frame=$j
        echo "Finished frame $j fold $i"
      done
    done
    
    # todo
    
    echo "Saved predictions to $EXP_NAME.pkl";;
    
##################################################
#          Table 1 Experiment 5
##################################################
    "Table1Exp5")
    
    export EXP_NAME=multilabel_engbert_focal3
    echo "Running $EXP_NAME"

    for i in 0 1 2 3 4
    do
      python main.py --model_type=bertmultilabel \
                     --data_dir=$DATA_GLOBAL_DIR/$i \
                     --task_name=frame \
                     --model_name_or_path=bert-base-uncased \
                     --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i \
                     --cache_dir=$CACHE_GLOBAL_DIR/english \
                     --per_gpu_train_batch_size=4 \
                     --learning_rate=2e-5 \
                     --num_train_epochs=10.0 \
                     --max_seq_length=128 \
                     --loss=focal3 \
                     --exp_name=$EXP_NAME \
                     --do_lower_case \
                     --do_eval \
                     --set_weights \
                     --do_train \
                     --overwrite_output_dir \
                     --overwrite_cache
      echo "Finished fold $i"
    done

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                              --exp_name=$EXP_NAME
                              
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR

    echo "Saved predictions to $EXP_NAME.pkl";;
    
##################################################
#          Table 1 Experiment 6
##################################################
    "Table1Exp6")
    
    export EXP_NAME=multilabel_multibert_focal3
    echo "Running $EXP_NAME"

    for i in 0 1 2 3 4
    do
      python main.py --model_type=bertmultilabel \
                     --data_dir=$DATA_GLOBAL_DIR/$i \
                     --task_name=frame \
                     --model_name_or_path=bert-base-multilingual-cased \
                     --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i \
                     --cache_dir=$CACHE_GLOBAL_DIR/multilingual \
                     --per_gpu_train_batch_size=4 \
                     --learning_rate=2e-5 \
                     --num_train_epochs=10.0 \
                     --max_seq_length=128 \
                     --loss=focal3 \
                     --exp_name=$EXP_NAME \
                     --do_train \
                     --do_eval \
                     --overwrite_output_dir \
                     --set_weights \
                     --overwrite_cache
      echo "Finished fold $i"
    done

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                              --exp_name=$EXP_NAME
                              
    echo "Saved predictions to $EXP_NAME.pkl"                                                  
                              
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR;;
    
    
##################################################
#          Table 1 Experiment 7
##################################################
    "Table1Exp7")
    
    export EXP_NAME=multilabel_multibert_bce
    echo "Running $EXP_NAME"

    for i in 0 1 2 3 4
    do
      python main.py --model_type=bertmultilabel \
                     --data_dir=$DATA_GLOBAL_DIR/$i \
                     --task_name=frame \
                     --model_name_or_path=bert-base-multilingual-cased 
                     --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i \
                     --cache_dir=$CACHE_GLOBAL_DIR/multilingual \
                     --per_gpu_train_batch_size=4 \
                     --learning_rate=2e-5 \
                     --num_train_epochs=10.0 \
                     --max_seq_length=128 \
                     --loss=bce \
                     --exp_name=$EXP_NAME \
                     --do_train \
                     --do_eval \
                     --overwrite_output_dir \
                     --overwrite_cache
      echo "Finished fold $i"
    done

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                              --exp_name=$EXP_NAME
    
    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR ;;

    
    
##################################################
#          Table 2 Experiment 1
##################################################
    "Table2Exp1DE")
    export EXP_NAME=multilabel_multibert_german_zero_shot_focal3_train1300
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
                   --data_dir=$DATA_GLOBAL_DIR/german/TrainEnTestDe \
                   --task_name=frame \
                   --model_name_or_path=bert-base-multilingual-cased \
                   --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                   --max_seq_length=128 \
                   --loss=focal3 \
                   --exp_name=$EXP_NAME \
                   --do_eval \
                   --do_train \
                   --overwrite_cache \
                   --cache_dir=$CACHE_GLOBAL_DIR/multilingual \
                   --per_gpu_train_batch_size=4 \
                   --learning_rate=2e-5 \
                   --num_train_epochs=13.0 \
                   --set_weights \
                   --overwrite_output_dir

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/german/TrainEnTestDe \
                           --target
    ;;
    
    
    
    "Table2Exp1AR")
    
    export EXP_NAME=multilabel_multibert_cased_focal3_arabic_zero_shot
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
                    --data_dir=$DATA_GLOBAL_DIR/arabic \
                    --task_name=frame \
                    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_cased_focal3_train_en_test_de2en \
                    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                    --max_seq_length=128 \
                    --loss=focal3 \
                    --exp_name=$EXP_NAME \
                    --do_eval \
                    --zero_shot

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/arabic \
                           --target
    ;;
    

    
    "Table2Exp1TR")
    
    export EXP_NAME=multilabel_multibert_cased_focal3_turkish_zero_shot
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
                    --data_dir=$DATA_GLOBAL_DIR/turkish \
                    --task_name=frame \
                    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_cased_focal3_train_en_test_de2en \
                    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                    --max_seq_length=128 \
                    --loss=focal3 \
                    --exp_name=$EXP_NAME \
                    --do_eval \
                    --zero_shot

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                                --exp_name=$EXP_NAME \
                                --fold_number=0
    
    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/turkish \
                           --target
    
    ;;

    
##################################################
#          Table 2 Experiment 2
##################################################    

    "Table2Exp2DE")
    
    export EXP_NAME=multilabel_multibert_german_focal3_train_code_switched
    echo "Running $EXP_NAME"

      python main.py --model_type=bertmultilabel \
      --data_dir=$DATA_GLOBAL_DIR/german/CodeSwitched \
      --task_name=frame \
      --model_name_or_path=bert-base-multilingual-cased \
      --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
      --max_seq_length=128 \
      --loss=focal3 \
      --exp_name=$EXP_NAME \
      --do_eval \
      --do_train \
      --overwrite_cache \
      --per_gpu_train_batch_size=4 \
      --learning_rate=2e-5 \
      --num_train_epochs=13.0 \
      --overwrite_output_dir \
      --set_weights \
      --cache_dir=$CACHE_GLOBAL_DIR/multilingual

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0
    
    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/german/CodeSwitched \
                           --target

    ;;
    


    "Table2Exp2AR")
    
    export EXP_NAME=multilabel_multibert_arabic_focal3_train_code_switched
    echo "Running $EXP_NAME"

      python main.py --model_type=bertmultilabel \
      --data_dir=$DATA_GLOBAL_DIR/arabic/CodeSwitched \
      --task_name=frame \
      --model_name_or_path=bert-base-multilingual-cased \
      --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
      --max_seq_length=128 \
      --loss=focal3 \
      --exp_name=$EXP_NAME \
      --do_eval \
      --overwrite_cache \
      --do_train \
      --per_gpu_train_batch_size=4 \
      --learning_rate=2e-5 \
      --num_train_epochs=13.0 \
      --max_seq_length=128 \
      --overwrite_output_dir \
      --set_weights \
      --cache_dir=$CACHE_GLOBAL_DIR/multilingual

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0
    
    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/arabic/CodeSwitched \
                           --target

    ;;
    
    
    
    "Table2Exp2TR")
    export EXP_NAME=multilabel_multibert_turkish_focal3_train_code_switched
    echo "Running $EXP_NAME"

      python main.py --model_type=bertmultilabel \
      --data_dir=$DATA_GLOBAL_DIR/turkish/CodeSwitched \
      --task_name=frame \
      --model_name_or_path=bert-base-multilingual-cased \
      --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
      --max_seq_length=128 \
      --loss=focal3 \
      --exp_name=$EXP_NAME \
      --do_eval \
      --overwrite_cache \
      --do_train \
      --per_gpu_train_batch_size=4 \
      --learning_rate=2e-5 \
      --num_train_epochs=13.0 \
      --overwrite_output_dir \
      --set_weights \
      --cache_dir=$CACHE_GLOBAL_DIR/multilingual

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/turkish/CodeSwitched \
                           --target
    ;;
    
##################################################
#          Table 2 Experiment 3
##################################################   

    "Table2Exp3DE")
    
    export EXP_NAME=multilabel_multibert_cased_focal3_german_few_shot
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
    --data_dir=$DATA_GLOBAL_DIR/german/FewShot \
    --task_name=frame \
    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_german_zero_shot_focal3_train1300 \
    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --max_seq_length=128 \
    --loss=focal3 \
    --exp_name=$EXP_NAME \
    --do_eval \
    --do_train \
    --per_gpu_train_batch_size=4 \
    --learning_rate=2e-5 \
    --num_train_epochs=13.0 \
    --overwrite_output_dir \
    --overwrite_cache \
    --set_weights

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0
    
    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/german/FewShot \
                           --target

    ;;


    
    "Table2Exp3AR")
    
    export EXP_NAME=multilabel_multibert_cased_focal3_arabic_few_shot
    echo "Running $EXP_NAME"
    
    python main.py --model_type=bertmultilabel \
    --data_dir=$DATA_GLOBAL_DIR/arabic/FewShot \
    --task_name=frame \
    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_cased_focal3_train_en_test_de2en \
    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --max_seq_length=128 \
    --loss=focal3 \
    --exp_name=$EXP_NAME \
    --do_eval \
    --do_train \
    --per_gpu_train_batch_size=4 \
    --learning_rate=2e-5 \
    --num_train_epochs=13.0 \
    --set_weights \
    --overwrite_output_dir \
    --overwrite_cache
    
    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0
    
    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/arabic/FewShot \
                           --target

    ;;
    
    
    
    "Table2Exp3TR")
    
    export EXP_NAME=multilabel_multibert_cased_focal3_turkish_few_shot
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
    --data_dir=$DATA_GLOBAL_DIR/turkish/FewShot \
    --task_name=frame \
    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_cased_focal3_train_en_test_de2en \
    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --max_seq_length=128 \
    --loss=focal3 \
    --exp_name=$EXP_NAME \
    --do_eval \
    --do_train \
    --per_gpu_train_batch_size=4 \
    --learning_rate=2e-5 \
    --num_train_epochs=10.0 \
    --set_weights \
    --overwrite_output_dir \
    --overwrite_cache

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME 
    --exp_name=$EXP_NAME 
    --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/turkish/FewShot \
                           --target
    ;;
    
    
    
##################################################
#          Table 2 Experiment 4
##################################################    

    "Table2Exp4DE")
    export EXP_NAME=multilabel_multibert_german_focal3_train_code_switched_few_shot
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
    --data_dir=$DATA_GLOBAL_DIR/german/FewShot \
    --task_name=frame \
    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_german_focal3_train_code_switched \
    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --max_seq_length=128 \
    --loss=focal3 \
    --exp_name=$EXP_NAME \
    --do_eval \
    --overwrite_cache \
    --do_train \
    --per_gpu_train_batch_size=4 \
    --learning_rate=2e-5 \
    --num_train_epochs=13.0 --max_seq_length=128 \
    --overwrite_output_dir \
    --cache_dir=$CACHE_GLOBAL_DIR/multilingual \
    --set_weights

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/german/FewShot \
                           --target
    
    ;;



    "Table2Exp4AR")
    export EXP_NAME=multilabel_multibert_arabic_focal3_train_code_switched_few_shot
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
    --data_dir=$DATA_GLOBAL_DIR/arabic/FewShot \
    --task_name=frame \
    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_arabic_focal3_train_code_switched \
    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --max_seq_length=128 \
    --loss=focal3 \
    --exp_name=$EXP_NAME \
    --do_eval \
    --overwrite_cache \
    --do_train \
    --per_gpu_train_batch_size=4 \
    --learning_rate=2e-5 \
    --num_train_epochs=13.0 \
    --overwrite_output_dir \
    --set_weights \
    --cache_dir=$CACHE_GLOBAL_DIR/multilingual

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/arabic/FewShot \
                           --target
    ;;



    "Table2Exp4TR")
    export EXP_NAME=multilabel_multibert_turkish_focal3_train_code_switched_few_shot
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
    --data_dir=$DATA_GLOBAL_DIR/turkish/FewShot \
    --task_name=frame \
    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_turkish_focal3_train_code_switched \
    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --max_seq_length=128 \
    --loss=focal3 \
    --exp_name=$EXP_NAME \
    --do_eval \
    --overwrite_cache \
    --do_train \
    --per_gpu_train_batch_size=4 \
    --learning_rate=2e-5 \
    --num_train_epochs=10.0 \
    --overwrite_output_dir \
    --set_weights \
    --cache_dir=$CACHE_GLOBAL_DIR/multilingual

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/turkish/FewShot \
                           --target
    
    ;;
    
    
##################################################
#          Table 3 Experiment 1
##################################################   
    "Table3Exp1DE")
    export EXP_NAME=multilabel_multibert_focal3_train_en2de_test_de
    echo "Running $EXP_NAME"

    python main.py  --model_type=bertmultilabel \
                    --data_dir=$DATA_GLOBAL_DIR/german/TrainEn2DeTestDe \
                    --task_name=frame \
                    --model_name_or_path=bert-base-multilingual-cased \
                    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                    --cache_dir=$CACHE_GLOBAL_DIR/multilingual \
                    --max_seq_length=128 \
                    --loss=focal3 \
                    --exp_name=$EXP_NAME \
                    --do_eval \
                    --do_train \
                    --per_gpu_train_batch_size=4 \
                    --learning_rate=2e-5 \
                    --num_train_epochs=13.0 \
                    --set_weights \
                    --overwrite_output_dir \
                    --overwrite_cache

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                              --exp_name=$EXP_NAME \
                              --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/german/TrainEn2DeTestDe \
                           --target
    ;;
    
    
    
    "Table3Exp1AR")
    export EXP_NAME=multilabel_multibert_focal3_train_en2ar_test_ar
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
                   --data_dir=$DATA_GLOBAL_DIR/arabic/TrainEn2ArTestAr \
                   --task_name=frame \
                   --model_name_or_path=bert-base-multilingual-cased \
                   --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                   --cache_dir=$CACHE_GLOBAL_DIR/multilingual \
                   --max_seq_length=128 \
                   --loss=focal3 \
                   --exp_name=$EXP_NAME \
                   --do_eval \
                   --do_train \
                   --per_gpu_train_batch_size=4 \
                   --learning_rate=2e-5 \
                   --num_train_epochs=13.0 \
                   --set_weights \
                   --overwrite_output_dir \
                   --overwrite_cache

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                              --exp_name=$EXP_NAME \
                              --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/arabic/TrainEn2ArTestAr \
                           --target
    
    ;;
    
    
    
    "Table3Exp1TR")
    export EXP_NAME=multilabel_multibert_focal3_train_en2tr_test_tr
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
                   --data_dir=$DATA_GLOBAL_DIR/turkish/TrainEn2TrTestTr \
                   --task_name=frame \
                   --model_name_or_path=bert-base-multilingual-cased \
                   --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                   --cache_dir=$CACHE_GLOBAL_DIR/multilingual \
                   --max_seq_length=128 \
                   --loss=focal3 \
                   --exp_name=$EXP_NAME \
                   --do_eval \
                   --do_train \
                   --per_gpu_train_batch_size=4 \
                   --learning_rate=2e-5 \
                   --num_train_epochs=13.0 \
                   --set_weights \
                   --overwrite_output_dir \
                   --overwrite_cache

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                              --exp_name=$EXP_NAME \
                              --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/turkish/TrainEn2TrTestTr \
                           --target
    
    ;;
    

##################################################
#          Table 3 Experiment 2
################################################## 
    "Table3Exp2DE")
    export EXP_NAME=multilabel_multibert_cased_focal3_train_en_test_de2en
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
                   --data_dir=$DATA_GLOBAL_DIR/german/TrainEnTestDe2En \
                   --task_name=frame \
                   --model_name_or_path=bert-base-multilingual-cased \
                   --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                   --cache_dir=$CACHE_GLOBAL_DIR/multiligual \
                   --max_seq_length=128 \
                   --loss=focal3 \
                   --exp_name=$EXP_NAME \
                   --do_eval \
                   --do_train \
                   --per_gpu_train_batch_size=4 \
                   --learning_rate=2e-5 \
                   --num_train_epochs=11.0 \
                   --set_weights \
                   --overwrite_output_dir \
                   --overwrite_cache

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                              --exp_name=$EXP_NAME \
                              --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/german/TrainEnTestDe2En \
                           --target
    
    ;;   
    
    
    
    
    "Table3Exp2AR")

    export EXP_NAME=multilabel_multibert_cased_focal3_train_en_test_ar2en
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
    --data_dir=$DATA_GLOBAL_DIR/arabic/TrainEnTestAr2En \
    --task_name=frame \
    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_cased_focal3_train_en_test_de2en \
    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --max_seq_length=128 \
    --loss=focal3 \
    --exp_name=$EXP_NAME \
    --do_eval \
    --zero_shot


    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/arabic/TrainEnTestAr2En \
                           --target
    ;;    



    "Table3Exp2TR")
    
    export EXP_NAME=multilabel_multibert_cased_focal3_train_en_test_tr2en
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
    --data_dir=$DATA_GLOBAL_DIR/turkish/TrainEnTestTr2En \
    --task_name=frame \
    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_cased_focal3_train_en_test_de2en \
    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --max_seq_length=128 \
    --loss=focal3 \
    --exp_name=$EXP_NAME \
    --do_eval \
    --zero_shot

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/turkish/TrainEnTestTr2En\
                           \
                           --target
    
    ;;
    
    
##################################################
#          Table 3 Experiment 3
################################################## 
    "Table3Exp3DE")
    
export EXP_NAME=multilabel_engbert_uncased_focal3_train_en_test_de2en
echo "Running $EXP_NAME"

python main.py --model_type=bertmultilabel \
--data_dir=$DATA_GLOBAL_DIR/german/TrainEnTestDe2En \
--task_name=frame \
--model_name_or_path=bert-base-uncased \
--output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
--cache_dir=$CACHE_GLOBAL_DIR/english \
--max_seq_length=128 \
--loss=focal3 \
--exp_name=$EXP_NAME \
--do_eval \
--do_train \
--per_gpu_train_batch_size=4 \
--learning_rate=2e-5 \
--num_train_epochs=13.0 \
--set_weights \
--do_lower_case

python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
--exp_name=$EXP_NAME \
--fold_number=0

echo "Saved predictions to $EXP_NAME.pkl"

python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/german/TrainEnTestDe2En \
                           --target

;;    



    "Table3Exp3AR")
    
export EXP_NAME=multilabel_engbert_uncased_focal3_train_en_test_ar2en_few_shot
echo "Running $EXP_NAME"

python main.py --model_type=bertmultilabel \
--data_dir=$DATA_GLOBAL_DIR/arabic/FewShotTranslated2En \
--task_name=frame \
--model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_engbert_uncased_focal3_train_en_test_de2en \
--output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
--max_seq_length=128 \
--loss=focal3 \
--exp_name=$EXP_NAME \
--do_eval \
--do_lower_case \
--do_train \
--per_gpu_train_batch_size=4 \
--learning_rate=2e-5 \
--num_train_epochs=13.0 \
--set_weights \
--overwrite_output_dir \
--overwrite_cache

python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
--exp_name=$EXP_NAME \
--fold_number=0

echo "Saved predictions to $EXP_NAME.pkl"

python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/arabic/FewShotTranslated2En \
                           --target
;;




    "Table3Exp3TR")
    
export EXP_NAME=multilabel_engbert_uncased_focal3_train_en_test_tr2en
echo "Running $EXP_NAME"

python main.py --model_type=bertmultilabel \
--data_dir=$DATA_GLOBAL_DIR/turkish/TrainEnTestTr2En \
--task_name=frame \
--model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_engbert_uncased_focal3_train_en_test_de2en \
--output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
--max_seq_length=128 \
--loss=focal3 \
--exp_name=$EXP_NAME \
--do_eval \
--do_lower_case \
--zero_shot

python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
--exp_name=$EXP_NAME \
--fold_number=0

echo "Saved predictions to $EXP_NAME.pkl"

python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/turkish/TrainEnTestTr2En \
                           --target
;;

##################################################
#          Table 3 Experiment 4
################################################## 



    "Table3Exp4DE")
    
export EXP_NAME=multilabel_engbert_cased_focal3_train_en_test_de2en
echo "Running $EXP_NAME"

python main.py --model_type=bertmultilabel \
--data_dir=$DATA_GLOBAL_DIR/german/TrainEnTestDe2En \
--task_name=frame \
--model_name_or_path=bert-base-cased \
--output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
--cache_dir=$CACHE_GLOBAL_DIR/english \
--max_seq_length=128 \
--loss=focal3 \
--exp_name=$EXP_NAME \
--do_eval \
--do_train \
--per_gpu_train_batch_size=4 \
--learning_rate=2e-5 \
--num_train_epochs=13.0 \
--set_weights

python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
--exp_name=$EXP_NAME \
--fold_number=0

echo "Saved predictions to $EXP_NAME.pkl"

python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/german/TrainEnTestDe2En \
                           --target
;;




    "Table3Exp4AR")
    export EXP_NAME=multilabel_engbert_cased_focal3_train_en_test_ar2en
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
    --data_dir=$DATA_GLOBAL_DIR/arabic/TrainEnTestAr2En \
    --task_name=frame \
    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_engbert_cased_focal3_train_en_test_de2en \
    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --max_seq_length=128 \
    --loss=focal3 \
    --exp_name=$EXP_NAME \
    --do_eval \
    --zero_shot

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"

    python run_analysis.py --exp_name=$EXP_NAME \
                               --thres=sigmoid \
                               --data_path=$DATA_GLOBAL_DIR/arabic/TrainEnTestAr2En \
                               --target
;;



    "Table3Exp4TR")
    export EXP_NAME=multilabel_engbert_cased_focal3_train_en_test_tr2en
    echo "Running $EXP_NAME"


    python main.py --model_type=bertmultilabel \
    --data_dir=$DATA_GLOBAL_DIR/turkish/TrainEnTestTr2En \
    --task_name=frame \
    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_engbert_cased_focal3_train_en_test_de2en \
    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --max_seq_length=128 \
    --loss=focal3 \
    --exp_name=$EXP_NAME \
    --do_eval \
    --zero_shot

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"

    python run_analysis.py --exp_name=$EXP_NAME \
                               --thres=sigmoid \
                               --data_path=$DATA_GLOBAL_DIR/turkish/TrainEnTestTr2En \
                               --target

;;


##################################################
#          Table 3 Experiment 5
################################################## 

    "Table3ExpDE")
    export EXP_NAME=multilabel_engbert_uncased_focal3_train_en_test_de2en_few_shot
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
    --data_dir=$DATA_GLOBAL_DIR/german/FewShotTranslated2En \
    --task_name=frame \
    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_engbert_uncased_focal3_train_en_test_de2en \
    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --max_seq_length=128 \
    --loss=focal3 \
    --exp_name=$EXP_NAME \
    --do_eval \
    --do_lower_case \
    --do_train \
    --per_gpu_train_batch_size=4 \
    --learning_rate=2e-5 \
    --num_train_epochs=13.0 \
    --set_weights \
    --overwrite_output_dir \
    --overwrite_cache

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/german/FewShotTranslated2En \
                           --target
    ;;    



    "Table3Exp5AR")
    export EXP_NAME=multilabel_engbert_uncased_focal3_train_en_test_ar2en_few_shot
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
    --data_dir=$DATA_GLOBAL_DIR/arabic/FewShotTranslated2En \
    --task_name=frame \
    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_engbert_uncased_focal3_train_en_test_de2en \
    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --max_seq_length=128 \
    --loss=focal3 \
    --exp_name=$EXP_NAME \
    --do_eval \
    --do_lower_case \
    --do_train \
    --per_gpu_train_batch_size=4 \
    --learning_rate=2e-5 \
    --num_train_epochs=13.0 \
    --set_weights \
    --overwrite_output_dir \
    --overwrite_cache

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/arabic/FewShotTranslated2En \
                           --target
    ;;



    "Table3Exp5TR")
    export EXP_NAME=multilabel_multibert_cased_focal3_train_en_test_tr2en_few_shot
    echo "Running $EXP_NAME"


    python main.py --model_type=bertmultilabel \
    --data_dir=$DATA_GLOBAL_DIR/turkish/FewShotTranslated2En \
    --task_name=frame \
    --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_cased_focal3_train_en_test_de2en \
    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --max_seq_length=128 --loss=focal3 \
    --exp_name=$EXP_NAME \
    --do_eval \
    --do_train \
    --per_gpu_train_batch_size=4 \
    --learning_rate=2e-5 \
    --num_train_epochs=13.0 \
    --set_weights \
    --overwrite_output_dir \
    --overwrite_cache


    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/turkish/FewShotTranslated2En \
                           --target
    ;;
    
    
##################################################
#          Table 4 Experiment 1
################################################## 

# Experiment 1 in this Table is the same as Experiment 1 in Table 2 for DE.

##################################################
#          Table 4 Experiment 2
################################################## 

    "Table4Exp2DE")
    export EXP_NAME=multilabel_multibert_german_focal3_train_omitted_code_switched
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
    --data_dir=$DATA_GLOBAL_DIR/german/OmittedCodeSwitch \
    --task_name=frame \
    --model_name_or_path=bert-base-multilingual-cased \
    --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --max_seq_length=128 \
    --loss=focal3 \
    --exp_name=$EXP_NAME \
    --do_eval \
    --do_train \
    --overwrite_cache \
    --per_gpu_train_batch_size=4 \
    --learning_rate=2e-5 \
    --num_train_epochs=18.0 \
    --overwrite_output_dir \
    --set_weights \
    --cache_dir=$CACHE_GLOBAL_DIR/multilingual \
    --eval_specific_checkpoint=checkpoint-3500

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
    --exp_name=$EXP_NAME \
    --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/german/OmittedCodeSwitch \
                           --target
    ;;
    
##################################################
#          Table 4 Experiment 3
################################################## 

# Experiment 3 in this Table is the same as Experiment 2 in Table 2 for DE.

##################################################
#          Table 4 Experiment 4
################################################## 

    "Table4Exp4DE")
    export EXP_NAME=multilabel_multibert_german_focal3_train_omitted_npmi_code_switched
    echo "Running $EXP_NAME"

    python main.py --model_type=bertmultilabel \
                   --data_dir=$DATA_GLOBAL_DIR/german/OmittednPMICodeSwitch \
                   --task_name=frame \
                   --model_name_or_path=bert-base-multilingual-cased \
                   --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                   --max_seq_length=128 \
                   --loss=focal3 \
                   --exp_name=$EXP_NAME \
                   --do_eval \
                   --do_train \
                   --overwrite_cache \
                   --per_gpu_train_batch_size=4 \
                   --learning_rate=2e-5 \
                   --num_train_epochs=18.0 \
                   --overwrite_output_dir \
                   --set_weights \
                   --save_steps=200 \
                   --cache_dir=$CACHE_GLOBAL_DIR/multilingual \
                   --eval_specific_checkpoint=checkpoint-3600

    python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME \
                              --exp_name=$EXP_NAME \
                              --fold_number=0

    echo "Saved predictions to $EXP_NAME.pkl"
    
    python run_analysis.py --exp_name=$EXP_NAME \
                           --thres=sigmoid \
                           --data_path=$DATA_GLOBAL_DIR/german/OmittednPMICodeSwitch  \
                           --target
    ;;
    
    
    *)
    echo "Experiment name $EXPERIMENT not understood."
    ;;
    
esac