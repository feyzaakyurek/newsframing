#! /bin/sh

export OUTPUT_GLOBAL_DIR=/projectnb/llamagrp/feyzanb
export DATA_GLOBAL_DIR=/project/llamagrp/feyza/transformers/dataset
export CACHE_GLOBAL_DIR=/projectnb/llamagrp/feyzanb/.cache/newsframing

# --------------------------------------------------------------------------

export EXP_NAME=multilabel_engbert_focal
echo "Running $EXP_NAME"

for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/$i --task_name=frame --model_name_or_path=bert-base-uncased --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/english --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=focal --exp_name=$EXP_NAME --do_lower_case --do_eval --set_inverse_normed_freqs --do_train --overwrite_output_dir
  echo "Finished fold $i"
done
  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

# --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_focal
echo "Running $EXP_NAME"

for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/$i --task_name=frame --model_name_or_path=bert-base-multilingual-cased --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/multilingual --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=focal --exp_name=$EXP_NAME --do_train --do_eval --overwrite_output_dir --set_inverse_normed_freqs
  echo "Finished fold $i"
done
  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

# --------------------------------------------------------------------------

export EXP_NAME=multiclass_engbert_softmaxfocal
echo "Running $EXP_NAME"

for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/dataset_multiclass_wide/$i --task_name=frame --model_name_or_path=bert-base-uncased --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/english --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=softmax-focal --exp_name=$EXP_NAME --do_train --do_eval --do_lower_case --overwrite_output_dir --set_inverse_normed_freqs
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

# # --------------------------------------------------------------------------

export EXP_NAME=multiclass_multibert_softmaxfocal
echo "Running $EXP_NAME"

for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/dataset_multiclass_wide/$i --task_name=frame --model_name_or_path=bert-base-multilingual-cased --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/multilingual --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=softmax-focal --exp_name=$EXP_NAME --do_train --do_eval --overwrite_output_dir --set_inverse_normed_freqs
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"
# # --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_softmax
echo "Running $EXP_NAME"

for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/$i --task_name=frame --model_name_or_path=bert-base-multilingual-cased --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/multilingual --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=softmax --exp_name=$EXP_NAME --do_train --do_eval --overwrite_output_dir
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

# --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_softmax_weighted
echo "Running $EXP_NAME"

for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/$i --task_name=frame --model_name_or_path=bert-base-multilingual-cased --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/multilingual --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=softmax-weighted --exp_name=$EXP_NAME --do_train --do_eval --overwrite_output_dir --set_inverse_normed_freqs
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

# --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_normalized_log_softmax
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/$i --task_name=frame --model_name_or_path=bert-base-multilingual-cased --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/multilingual --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=normalized-log-softmax --exp_name=$EXP_NAME --do_train --do_eval --overwrite_output_dir
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"


# --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_log_normalized_softmax
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/$i --task_name=frame --model_name_or_path=bert-base-multilingual-cased --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/multilingual --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=log-normalized-softmax --exp_name=$EXP_NAME --do_train --do_eval --overwrite_output_dir
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"
# --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_german_zero_shot_focal
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german --task_name=frame --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_focal/$i --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --max_seq_length=128 --loss=focal --exp_name=$EXP_NAME --do_eval --zero_shot
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"
# --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_german_zero_shot_softmax
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german --task_name=frame --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_softmax/$i --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --max_seq_length=128 --loss=softmax --exp_name=$EXP_NAME --do_eval --zero_shot
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

# # --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_german_zero_shot_softmax_weighted
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german --task_name=frame --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_softmax_weighted/$i --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --max_seq_length=128 --loss=softmax-weighted --exp_name=$EXP_NAME --do_eval --zero_shot
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

# --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_german_30_shot_focal
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german/30shot --task_name=frame --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_focal/$i --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=focal --exp_name=$EXP_NAME --do_eval --do_train --overwrite_output_dir
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"
# # --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_german_30_shot_softmax
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german/30shot --task_name=frame --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_softmax/$i --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=softmax --exp_name=$EXP_NAME --do_eval --do_train --overwrite_output_dir
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

# # --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_german_30_shot_softmax_weighted
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german/30shot --task_name=frame --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_softmax_weighted/$i --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i ------per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=softmax-weighted --exp_name=$EXP_NAME --do_eval --do_train --overwrite_output_dir --set_inverse_normed_freqs
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

# --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_focal_train_en2de_test_de
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german/TrainEnglish2German/$i --task_name=frame --model_name_or_path=bert-base-multilingual-cased --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/multilingual --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=focal --exp_name=$EXP_NAME --do_eval --do_train --overwrite_output_dir --set_inverse_normed_freqs
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"
# # --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_softmax_train_en2de_test_de
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german/TrainEnglish2German/$i --task_name=frame --model_name_or_path=bert-base-multilingual-cased --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/multilingual --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=softmax --exp_name=$EXP_NAME --do_eval --do_train --overwrite_output_dir
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"    

# # --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_softmax_weighted_train_en2de_test_de
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german/TrainEnglish2German/$i --task_name=frame --model_name_or_path=bert-base-multilingual-cased --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/multilingual --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=softmax-weighted --exp_name=$EXP_NAME --do_eval --do_train --overwrite_output_dir --set_inverse_normed_freqs
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"  


# # --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_focal_train_en_test_de2en
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german/TestGerman2English --task_name=frame --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_focal/$i --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --max_seq_length=128 --loss=focal --exp_name=$EXP_NAME --do_eval --zero_shot
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"
# --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_softmax_train_en_test_de2en
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german/TestGerman2English --task_name=frame --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_softmax/$i --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --max_seq_length=128 --loss=softmax --exp_name=$EXP_NAME --do_eval --zero_shot
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

# --------------------------------------------------------------------------

export EXP_NAME=multilabel_multibert_softmax_weighted_train_en_test_de2en
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german/TestGerman2English --task_name=frame --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_multibert_softmax_weighted/$i --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --max_seq_length=128 --loss=softmax-weighted --exp_name=$EXP_NAME --do_eval --zero_shot
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

# --------------------------------------------------------------------------

export EXP_NAME=multilabel_engbert_focal_train_en_test_de2en
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german/TestGerman2English --task_name=frame --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_engbert_focal/$i --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/english --max_seq_length=128 --loss=focal --exp_name=$EXP_NAME --do_eval --do_lower_case --zero_shot
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

--------------------------------------------------------------------------

export EXP_NAME=multilabel_engbert_cased_focal
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/$i --task_name=frame --model_name_or_path=bert-base-cased --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/english_cased --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=focal --exp_name=$EXP_NAME --do_eval --overwrite_cache --overwrite_output_dir --set_inverse_normed_freqs --do_train
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

# --------------------------------------------------------------------------

export EXP_NAME=multilabel_engbert_cased_focal_train_en_test_de2en
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german/TestGerman2English --task_name=frame --model_name_or_path=$OUTPUT_GLOBAL_DIR/multilabel_engbert_cased_focal/$i --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/english_cased --max_seq_length=128 --loss=focal --exp_name=$EXP_NAME --do_eval --zero_shot
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

# --------------------------------------------------------------------------

export EXP_NAME=multilabel_engbert_softmax_train_en_test_de2en
echo "Running $EXP_NAME"


for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german/TestGerman2English/$i --task_name=frame --model_name_or_path=bert-base-uncased --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/english --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=softmax --exp_name=$EXP_NAME --do_eval --do_lower_case --do_train --overwrite_output_dir
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

# --------------------------------------------------------------------------

export EXP_NAME=multilabel_engbert_softmax_weighted_train_en_test_de2en

for i in 0 1 2 3 4
do
  python main.py --model_type=bertmultilabel --data_dir=$DATA_GLOBAL_DIR/german/TestGerman2English/$ --task_name=frame --model_name_or_path=bert-base-uncased --output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME/$i --cache_dir=$CACHE_GLOBAL_DIR/english --per_gpu_train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=10.0 --max_seq_length=128 --loss=softmax-weighted --exp_name=$EXP_NAME --do_eval --do_lower_case --do_train --overwrite_output_dir --set_inverse_normed_freqs
  echo "Finished fold $i"
done

  
python collate_pickles.py --exp_output_dir=$OUTPUT_GLOBAL_DIR/$EXP_NAME --exp_name=$EXP_NAME

echo "Saved predictions to $EXP_NAME.pkl"

echo "Finished!"
















