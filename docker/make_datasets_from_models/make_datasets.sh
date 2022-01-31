# clone optimistic repo
git clone https://github.com/carismoses/optimistic.git

# link other packages
cd optimistic
ln -s /pb_robot/src/pb_robot .
ln -s /pddlstream/pddlstream .

# move models to right place
cp -r /$PATH1 logs/experiments/
cp -r /$PATH2 logs/experiments/
cp -r /$PATH3 logs/experiments/
cp -r /$PATH4 logs/experiments/
cp -r /$PATH5 logs/experiments/

# run training code
python3 -m experiments.gen_dataset \
            --max-actions $N_ACTIONS \
            --balanced $BALANCED \
            --exp-name $EXP_NAME \
            --goal-progress $GP \
            --n-datasets $N_DATASETS \
            --data-collection-mode $MODE \
            --model-paths logs/experiments/$PATH1 \
                          logs/experiments/$PATH2 \
                          logs/experiments/$PATH3 \
                          logs/experiments/$PATH4 \
                          logs/experiments/$PATH5
