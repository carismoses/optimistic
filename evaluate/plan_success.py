import argparse
import numpy as np
import matplotlib.pyplot as plt
from learning.utils import model_forward
from experiments.utils import ExperimentLogger
from domains.tools.world import ToolsWorld
from domains.utils import init_world

## Params
all_model_paths = {#'sequential-goals-val-and-init': ['logs/experiments/sequential-goals-20220216-182615',
                #                        'logs/experiments/sequential-goals-20220216-183036',
                #                        'logs/experiments/sequential-goals-20220216-183101'],
                   'sequential-goals-init': ['logs/experiments/sequential-goals-noval-20220216-205218',
                                        'logs/experiments/sequential-goals-noval-20220216-205236',
                                        'logs/experiments/sequential-goals-noval-20220217-040812'],
                   'sequential-goals-old': ['logs/experiments/sequential-goals-20220208-025351',
                                        'logs/experiments/sequential-goals-20220208-025356',
                                        'logs/experiments/sequential-goals-20220208-025405'],
                   #'random-actions': ['logs/experiments/random-actions-20220207-192035',
                    #                    'logs/experiments/random-actions-20220207-205436',
                    #                    'logs/experiments/random-actions-20220207-223028'],
                   #'random-goals': ['logs/experiments/random-goals-opt-20220207-192020',
                    #                    'logs/experiments/random-goals-opt-20220207-204938',
                    #                    'logs/experiments/random-goals-opt-20220207-221248'],
                   #'sequential-actions': ['logs/experiments/sequential-plans-20220208-011124',
                    #                    'logs/experiments/sequential-plans-20220208-141044',
                    #                    'logs/experiments/sequential-plans-20220208-230530'],
                   }

def calc_p_success(trajectories):
    success = []
    for trajectory in trajectories:
        if len(trajectory) == 0: success.append(0.0)
        else: success.append(float(all([t[-1] for t in trajectory])))
    return sum(success)/len(success)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    parser.add_argument('--actions-step',
                        type=int,
                        help='number of actions between each datapoint')
    parser.add_argument('--single-action-step',
                        type=int,
                        help='use if want to just calculate plan success for a single action step')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    fig, ax = plt.subplots()
    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for pi, (method, model_paths) in enumerate(all_model_paths.items()):
        print('Method: %s' % method)
        all_successes = []
        for mi, model_path in enumerate(model_paths):
            print('Model path: %s' % model_path)
            model_logger = ExperimentLogger(model_path)
            if args.actions_step:
                assert not args.actions_step % model_logger.args.train_freq, \
                            'actions-step arg must be divisible by train_freq'
                successes = []
                n_actions = []
                for trajectories, ti in model_logger.get_trajectories_iterator():
                    p_success = calc_p_success(trajectories)
                    successes.append(p_success)
                    n_actions.append(ti)
                all_successes.append(successes)

                condition = len(set([len(success_list) for success_list in all_successes])) == 1
                assert condition, 'all models for method %s do not contain the same amount of actions/models' % method
                all_successes = np.array(all_successes)
                avg_success = np.mean(all_successes, axis=0)
                std_success = np.std(all_successes, axis=0)
                ax.plot(n_actions, avg_success, color=cs[pi % len(cs)], label=method)
                ax.fill_between(n_actions, avg_success-std_success, avg_success+std_success, color=cs[pi % len(cs)], alpha=0.1)
            if args.single_action_step:
                trajectories = model_logger.load_trajectories(args.single_action_step)
                p_success = calc_p_success(trajectories)
                print(model_path, p_success)

    if args.actions_step:
        ax.set_xlabel('Number of Executed Actions')
        ax.set_ylabel('% Plan Execution Success')
        ax.set_title('Plan Execution Success over Training Time')
        ax.legend(title='Method')
        ax.set_ylim([0,1])
        plt.savefig('plan_success.svg', format='svg')
        #plt.show()
