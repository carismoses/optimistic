# configure minio
#./mc config host add honda_cmm $CSAIL_ENDPOINT $HONDA_ACCESS $HONDA_SECRET

# clone optimistic repo
git clone https://github.com/carismoses/optimistic.git

# link other packages
cd optimistic
ln -s /pb_robot/src/pb_robot .
ln -s /pddlstream/pddlstream .

# run training code
python3 -m experiments.curriculum \
              --domain tools \
              --exp-name curriculum \
              --data-collection-mode curriculum \
              --max-actions 100 \
              --actions-per-curric 10 \
              --train-freq 2 \
              --n-models 1

# copy over results to minio
#/./mc cp -r learning/experiments/logs/ honda_cmm/stacking/rss_camera_ready/
