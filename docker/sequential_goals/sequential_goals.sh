# configure minio
#./mc config host add honda_cmm $CSAIL_ENDPOINT $HONDA_ACCESS $HONDA_SECRET

# clone optimistic repo
git clone https://github.com/carismoses/optimistic.git

# link other packages
cd optimistic
ln -s /pb_robot/src/pb_robot .
ln -s /pddlstream/pddlstream .

# run training code
python3 -m experiments.collect_data_and_train \
              --domain tools \
              --exp-name sequential_goals \
              --data-collection-mode sequential-goals \
              --n-models 7 \
              --n-seq-plans 100

# copy over results to minio
#/./mc cp -r learning/experiments/logs/ honda_cmm/stacking/rss_camera_ready/