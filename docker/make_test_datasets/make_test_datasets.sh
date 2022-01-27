# clone optimistic repo
git clone https://github.com/carismoses/optimistic.git

# link other packages
cd optimistic
ln -s /pb_robot/src/pb_robot .
ln -s /pddlstream/pddlstream .

# run training code
python3 -m evaluate.gen_test_dataset \
            --max-actions 100 \
            --exp-name test_dataset_progress0p0 \
            --goal-progress .0
