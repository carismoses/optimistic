import numpy as np
import matplotlib.pyplot as plt
from learning.utils import model_forward
from learning.utils import ExperimentLogger
from domains.tools.world import ToolsWorld
from domains.utils import init_world

## Params
all_model_paths = {'test': ['logs/experiments/test_eval-20220209-101243']
                }


if __name__ == '__main__':
    import pdb; pdb.set_trace()

    fig, ax = plt.subplots()
    cs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for pi, (method, model_paths) in enumerate(all_model_paths.items()):
        print('Method: %s' % method)
        all_successes = []
        for mi, model_path in enumerate(model_paths):
            print('Model path: %s' % model_path)
            successes = []
            n_actions = []
            model_logger = ExperimentLogger(model_path)
            for trajectories, ti in model_logger.get_trajectories_iterator():
                success = []
                for trajectory in trajectories:
                    if len(trajectory) == 0: success.append(0.0)
                    else: success.append(float(all([t[-1] for t in trajectory])))
                p_success = sum(success)/len(success)
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

    ax.set_xlabel('Number of Executed Actions')
    ax.set_ylabel('% Plan Execution Success')
    ax.set_title('Plan Execution Success over Training Time')
    ax.legend(title='Method')
    ax.set_ylim([0,1])
    plt.savefig('plan_success.svg', format='svg')
    #plt.show()
