# clone optimistic repo
git clone https://github.com/carismoses/optimistic.git

# link other packages
cd optimistic
ln -s /pb_robot/src/pb_robot .
ln -s /pddlstream/pddlstream .

# run training code
cp -r /$PATH1 logs/experiments/
cp -r /$PATH2 logs/experiments/
cp -r /$PATH3 logs/experiments/
cp -r /$PATH4 logs/experiments/
cp -r /$PATH5 logs/experiments/

python3 -m experiments.train_from_data \
            --dataset-exp-path logs/experiments/$PATH1 \
                               logs/experiments/$PATH2 \
                               logs/experiments/$PATH3 \
                               logs/experiments/$PATH4 \
                               logs/experiments/$PATH5 \
            --train-freq $TRAIN_FREQ
