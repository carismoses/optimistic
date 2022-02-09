from experiments.utils import ExperimentLogger
import matplotlib.pyplot as plt
import numpy as np

def plot_goal_analysis(dir, files_paths):
    #import pdb; pdb.set_trace()
    max_actions = 64
    all_plt_ys = []

    for paths in files_paths:
        plt_ys = []
        plt_xs = []
        logger = ExperimentLogger(paths)
        for dataset, di in logger.get_dataset_iterator():
            print(di)
            if di < max_actions:
                xs, ys = dataset[:]
                if len(ys) > 0:

                    percent_success = ys.tolist().count(1)/len(ys.tolist())
                    plt_ys.append(percent_success)
                    plt_xs.append(di)
                    print('-', len(plt_ys))
                    '''
                    x,y = dataset[-1]
                    action_vec = x[2]
                    final_pos = action_vec[:2]
                    dist = np.linalg.norm(final_pos - np.array([(0.4, -0.3)]))
                    plt_ys.append(dist)
                    plt_xs.append(di)
                    '''
        all_plt_ys.append(plt_ys)

    fig, ax = plt.subplots()
    all_plt_ys = np.array(all_plt_ys)
    avg_accs = np.mean(all_plt_ys, axis=0)
    std_accs = np.std(all_plt_ys, axis=0)
    xs = np.arange(len(plt_ys))
    ax.plot(plt_xs, avg_accs, color='r')
    ax.fill_between(plt_xs, avg_accs-std_accs, avg_accs+std_accs, color='r', alpha=0.1)
    plt.title('Push Distance over Time\n%s' % method)
    ax.set_xlabel('Num Actions')
    ax.set_ylabel('Push Distance for Actions')
    plt.savefig('%s_pushdist' % method)

dataset_paths = {#'random-actions':
                 #   ['logs/experiments/random_actions-20220104-201317',
                 #   'logs/experiments/random_actions-20220104-202422',
                 #   'logs/experiments/random_actions-20220104-202440',
                 #   'logs/experiments/random_actions-20220104-202447',
                 #   'logs/experiments/random_actions-20220104-202453'],
                #'random-goals-opt':
                #    ['logs/experiments/random_goals_opt-20220104-204547',
                #    'logs/experiments/random_goals_opt-20220104-203849',
                #    'logs/experiments/random_goals_opt-20220104-204627',
                #    'logs/experiments/random_goals_opt-20220104-204532',
                #    'logs/experiments/random_goals_opt-20220104-204536'],
                'sequential-goals':
                    ['logs/experiments/sequential_goals-20220120-030537',
                                    'logs/experiments/sequential_goals-20220120-030624',
                                    'logs/experiments/sequential_goals-20220120-030629',
                                    'logs/experiments/sequential_goals-20220120-030635',
                                    'logs/experiments/sequential_goals-20220120-030640'],
		#'engineered-goals-dist':
		#    ['logs/experiments/engineered_goals_dist-20220112-162941',
		#    'logs/experiments/engineered_goals_dist-20220112-162947',
		#    'logs/experiments/engineered_goals_dist-20220112-162956',
		#    'logs/experiments/engineered_goals_dist-20220112-163004',
		#    'logs/experiments/engineered_goals_dist-20220112-163058'],
		#'engineered-goals-size':
		#    ['logs/experiments/engineered_goals_size-20220112-172108',
		#    'logs/experiments/engineered_goals_size-20220112-172115',
		#    'logs/experiments/engineered_goals_size-20220112-172119',
		#    'logs/experiments/engineered_goals_size-20220112-172125',
 		#    'logs/experiments/engineered_goals_size-20220112-172129']
		}

for method, method_paths in dataset_paths.items():
    plot_goal_analysis(method, method_paths)
