# clone optimistic repo
git clone https://github.com/carismoses/optimistic.git

# link other packages
cd optimistic
ln -s /pb_robot/src/pb_robot .
ln -s /pddlstream/pddlstream .

# run training code

python3 -m experiments.train_from_data \
            --dataset-exp-path $PATH1 $PATH2 $PATH3 $PATH4 $PATH5 \
            --train-freq $TRAIN_FREQ
