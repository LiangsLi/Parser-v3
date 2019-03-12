#!/usr/bin/env bash
# Checking virtualenv
if [[ ! -f "env/bin/activate" ]];then
  echo "Please first build virtualenv with setup.sh!"
  exit
fi

# Preprocessing
if [[ ! -d "data/text-conllu" ]];then
  mkdir data/text-conllu
fi
# 获取输入参数
input=$1 #data/text-conll/text.test.conll

if [[ -z $1 || $1 == '-h' ]];then
  echo "usage./run_self.sh input-file output-directory"
  exit
fi

infile=`basename $input`
# 输入文件转换格式 sem16 -> conllu
output=data/$infile.conllu
echo "Preprocessing the conll file"
python converter/sem16_to_conllu.py $input $output

main=main.py
gpu=""
#save=saves/text-0
# 模型参数保存的位置
save=saves/sem16-v0 # E:/model_ckp/parser_v3_tf/sem16-v0
# 获取输出文件参数
out=$2
# $file是转换后的输入文件
file=$output
#other=saves/text-1:saves/text-2:saves/text-3:saves/text-4

# Parsing
if [[ ! -d outputs ]];then
  mkdir outputs
fi
source env/bin/activate
CUDA_VISIBLE_DEVICES=$gpu python3 $main --save_dir $save run --output_dir $out $file #--other_save_dirs $other

tosem16=converter/conllu_to_sem16.py
outfile=`basename $output`
# 最终输出转换为 sem16格式
python $tosem16 $out/$outfile
