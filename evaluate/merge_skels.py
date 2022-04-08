import dill as pickle

###
dirs = ['1', '2', '3']
skel_nums = [6, 7, 8]
###

path = 'logs/all_skels/skel%i/%s/ss_skeleton_samples.pkl'
dest_path = 'logs/all_skels/skel%i/ss_skeleton_samples.pkl'

import pdb; pdb.set_trace()
for skel_num in skel_nums:
    all_skels = {}
    for dir in dirs:
        skel_path = path % (skel_num, dir)
        print(skel_path)

        # load file
        with open(skel_path, 'rb') as f:
            path_skels = pickle.load(f)

        # print length
        key = list(path_skels.keys())[0]
        print(len(path_skels[key]))

        # make sure all same key
        if all_skels == {}:
            all_skels[key] = []
            all_skels_key = key
        else:
            all_skels_key = list(all_skels.keys())[0]
            if not ((all_skels_key.skeleton_fn.__name__ == key.skeleton_fn.__name__) and
                    (all_skels_key.goal_obj == key.goal_obj) and
                    (all_skels_key.ctypes == key.ctypes)):
                print('All given paths dont have matching keys\n    %s\n    %s' % (key, all_skels_key))
                break

        # add to main list
        all_skels[all_skels_key] += path_skels[key]

    # dump in dest path
    with open(dest_path%skel_num, 'wb') as f:
        pickle.dump(all_skels, f)
    print('Done with skel num %i!' % skel_num)

