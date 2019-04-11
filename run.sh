#!/usr/bin/env bash
# Checking virtualenv
#if [[ ! -f "env/bin/activate" ]];then
#  echo "Please first build virtualenv with setup.sh!"
#  exit
#fi

# Preprocessing
#if [[ ! -d "data/text-conllu" ]];then
#  mkdir data/text-conllu
#fi
input=data/text-conll/text.test.conll

#if [[ -z $1 || $1 == '-h' ]];then
#  echo "usage./run_self.sh input-file output-directory"
#  exit
#fi

infile=`basename $input`
output=data/$infile.conllu
echo "Preprocessing the conll file"
python converter/sem16_to_conllu.py $input $output

main=main.py
gpu=""
#save=saves/text-0
save=saves/sem16-v0
out=$2
file=$output
#other=saves/text-1:saves/text-2:saves/text-3:saves/text-4

# Parsing
if [[ ! -d outputs ]];then
  mkdir outputs
fi
source env/bin/activate
CUDA_VISIBLE_DEVICES=$gpu python3 $main --save_dir $save run --output_dir $out $file #--other_save_dirs $other

# python3 main.py --save_dir saves/sem16-v0 run --output_dir $2 data/$1.conllu

tosem16=converter/conllu_to_sem16.py
outfile=`basename $output`
python $tosem16 $out/$outfile
