cd $1_configs;
for config in *;
do
    echo "${config%%.*}";
    cd ../..;
    python3 experiment_runner.py --config_file "all_img_batch/$1_configs/${config}" --experiment_dir "experiments/$1/${config%%.*}" --mode completion;
    cd "all_img_batch/$1_configs";
done
