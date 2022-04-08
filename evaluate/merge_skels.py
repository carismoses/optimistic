import dill as pickle

###
dirs = ['1', '2', '3']
skel_nums = [6, 7, 8]
###

path = 'logs/all_skels/skel%i/%i/ss_skeleton_samples.pkl'
dest_path = 'logs/all_skels/skel%i/ss_skeleton_samples.pkl'

all_skels = {}

for skel_num in skel_nums:
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
        else:
            all_skels_key = list(all_skels.keys()[0])
            if all_skels_key not in all_skels:
                print('All given paths dont have matching keys\n    %s\n    %s') % (key, all_skels_key)
                break

        # add to main list
        all_skels[key].append(path_skels[key])

    # dump in dest path
    with open(dest_path, 'wb') as f:
        pickle.dump(all_skels f)
    print('Done with skel num %i!' % skel_num)
