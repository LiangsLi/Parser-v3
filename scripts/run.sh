main=/home/alex/work/codes/parser/Parser-v3/main.py
gpu=""
save=$1
out=$2
file=$3
#drop=$4
if [[ -z $2 || $1 == '-h' ]];then
  echo "usage:./run.sh (save dir) (output dir) (input file) [drop_arc]"
  exit
fi
if [[ ! -z $4 ]];then
  echo drop arc
  drop="--drop_arc "$4
fi

CUDA_VISIBLE_DEVICES=$gpu python3 $main --save_dir $save run --output_dir $out $file $drop

