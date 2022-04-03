import numpy as np
import matplotlib.pyplot as plt
from learning.utils import model_forward
from experiments.utils import ExperimentLogger
from domains.tools.world import ToolsWorld

## Params

all_model_paths = {'sequential-goals': ['sequential-goals-20220326-155023',
                                        'sequential-goals-20220326-154929',
                                        'sequential-goals-20220326-154919'],
                    'random-goals': ['random-goals-opt-20220326-155213',
                                        'random-goals-opt-20220326-155206',
                                        'random-goals-opt-20220326-155154']}

all_actions = ['pick', 'move_contact-poke', 'move_contact-push_pull', 'move_holding']
all_objs = ['yellow_block', 'blue_block']
max_actions = None

if __name__ == '__main__':
    import pdb; pdb.set_trace()
    fig, ax = plt.subplots()
    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for pi, (method, model_paths) in enumerate(all_model_paths.items()):
        print('Method: %s' % method)
        # calculate all plan successes
        method_successes = {}
        for mi, model_path in enumerate(model_paths):
            print('Path: %s' % model_path)
            logger = ExperimentLogger('logs/experiments/'+model_path)
            _, txs = logger.get_dir_indices('plan_success')
            for pii in sorted(txs):
                if max_actions is None or pii < max_actions:
                    pii_success_info = logger.load_success_data(pii)[1]
                    pii_success = []
                    for _,_,_,s in pii_success_info:
                        if isinstance(s, bool):
                            pii_success.append(s)
                        else:
                            pii_success.append(s[-1])

                    if pii in method_successes:
                        method_successes[pii] += pii_success
                    else:
                        method_successes[pii] = pii_success

                    # plot all goal and successes
                    sub_fig, sub_ax = plt.subplots()
                    for _, _, goal_pred, s in pii_success_info:
                        print(goal_pred)
                        if goal_pred is not None:
                            goal_xy = goal_pred[2].pose[0][:2]
                            goal_obj = goal_pred[1].readableName
                            su = s if isinstance(s, bool) else s[-1]
                            color = 'g' if su else 'r'
                            mc = 'y' if goal_obj == 'yellow_block' else 'b'
                            sub_ax.plot(*goal_xy, '.', color=color, markeredgecolor=mc)
                    fname = 'success_%i.svg' % (pii)
                    logger.save_figure(fname, dir='plan_success')
                    plt.close(sub_fig)

        # calculate avg and standard deviations
        method_avg = [np.mean(success_data) for success_data in method_successes.values()]
        method_std = [np.std(success_data) for success_data in method_successes.values()]

        # get max_actions and interpolate so all lists same length
        if not max_actions:
            max_actions = max(list(method_successes.keys()))
        xs = np.arange(max_actions)
        all_action_steps = list(method_successes.keys())
        full_mean = np.interp(xs, all_action_steps, method_avg)
        full_std = np.interp(xs, all_action_steps, method_std)

        # plot avg and standard dev
        ax.plot(xs, full_mean, color=cs[pi % len(cs)], label=method)
        ax.fill_between(xs, full_mean-full_std, full_mean+full_std, color=cs[pi % len(cs)], alpha=0.1)

    ax.set_xlabel('Number of Executed Actions')
    ax.set_ylabel('Plan Success')
    ax.set_title('Plan Success over Training Time')
    ax.legend(title='Method', loc='lower right')
    ax.set_ylim([0.,1])
    plt.savefig('plan_success.svg', format='svg')
    #plt.show()

