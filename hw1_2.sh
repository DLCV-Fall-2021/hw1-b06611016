#Download model
[ -f new_fcn8_best_model_hw1_2.ckpt?dl=1 ] && echo "Model exists" || wget 'https://www.dropbox.com/s/4m49j2dlpk094ex/new_fcn8_best_model_hw1_2.ckpt?dl=1'
#Execute the python file
python3 hw1_p2_evaluation.py $1 $2