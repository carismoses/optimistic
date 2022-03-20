import dill as pickle

if __name__ == '__main__':
    all_skels_path = 'logs/all_skels/skel'
    final_path = 'logs/ss_skeleton_samples.pkl'
    skel_nums = [0, 1, 2, 3, 6, 7, 8, 9]

    all_plans = []
    for skel_num in skel_nums:
        skel_path = '%s%i/ss_skeleton_samples.pkl' % (all_skels_path, skel_num)
        with open(skel_path, 'rb') as handle:
            skel_plans = pickle.load(handle)
        key = list(skel_plans.keys())[0]
        print('Adding skel %i: %s' % (skel_num, key))
        all_plans += skel_plans[key]
    print('Merged %s plans' % len(all_plans))
    with open(final_path, 'wb') as handle:
        pickle.dump(all_plans, handle)
