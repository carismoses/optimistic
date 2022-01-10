# configure minio
./mc config host add honda_cmm https://ceph.csail.mit.edu 03ba63de07cf423daa1bb0e4ecca35c9 bb6116aa4549415fb1c85fc2f03ffd34

# clone optimistic repo
git clone https://github.com/carismoses/optimistic.git

# move experiment to right path
mkdir -p optimistic/logs/experiments
mv sequential_goals-20220105-145239 optimistic/logs/experiments

# link other packages
cd optimistic
ln -s /pb_robot/src/pb_robot .
ln -s /pddlstream/pddlstream .

# run training code
python3 -m experiments.collect_data_and_train \
              --restart \
              --exp-path

# copy over results to minio
#/./mc cp -r learning/experiments/logs/ honda_cmm/stacking/rss_camera_ready/
