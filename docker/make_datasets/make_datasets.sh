# clone optimistic repo
git clone https://github.com/carismoses/optimistic.git

# link other packages
cd optimistic
ln -s /pb_robot/src/pb_robot .
ln -s /pddlstream/pddlstream .

# run training code

python3 -m experiments.gen_dataset \
            --max-actions $N_ACTIONS \
            --balanced $BALANCED \
            --exp-name $EXP_NAME \
            --goal-progress $GP \
            --n-datasets $N_DATASETS
