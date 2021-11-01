#Download model
[ -f new_best_model_hw1_1.ckpt?dl=1 ] && echo "Model exists" || wget 'https://www.dropbox.com/s/fxmssdwrl5brppc/new_best_model_hw1_1.ckpt?dl=1'
#Execute the python file
python3 hw1_p1_evaluation.py $1 $2