import time
import argparse
import matplotlib.pyplot as plt

from experiments.utils import ExperimentLogger
from domains.tools.world import ToolsWorld
from evaluate.plot_value_fns import get_model_accuracy_fn, get_seq_fn
from domains.tools.primitives import get_contact_gen


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    parser.add_argument('--exp-path',
                        type=str,
                        help='experiment path to visualize results for')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    logger = ExperimentLogger(args.exp_path)
    world = ToolsWorld(False, None, logger.args.actions, logger.args.objects)
    dir = 'dataset'
    dataset = logger.load_trans_dataset('')
    all_axes = {}
    ts = time.strftime('%Y%m%d-%H%M%S')
    if logger.args.goal_type == 'push':
        contacts_fn = get_contact_gen(world.panda.planning_robot, world.contact_types)
        contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)

        all_axes = {}
        for type, dataset in dataset.datasets.items():
            fig, axes = plt.subplots(2, figsize=(5,10))
            world.vis_dataset(axes[0], dataset, logger.args.goal_type)
            # add blue squares for goals that failed to plan
            failed_goals = logger.load_failed_plans()
            for _, contact_type, x, _ in failed_goals:
                if contact_type == type:
                    world.plot_block(axes[0], x, 'b')
            for contact in contacts:
                cont = contact[0]
                if cont.type == type:
                    world.vis_tool_ax(cont, axes[1], 'cont')

            axes[0].set_title('%s Dataset' % type)
            axes[1].set_title('Contact Configuration')

            axes[0].set_aspect('equal')
            axes[0].set_xlim([-1, 1])
            axes[0].set_ylim([-1, 1])

            all_axes[type] = axes

            fname = 'dataset_%s_%s.svg' % (ts, type)
            logger.save_figure(fname, dir=dir)
            plt.close()
    elif logger.args.goal_type == 'pick':
        fig, ax = plt.subplots()
        world.vis_dataset(ax, dataset, logger.args.goal_type)
        # add blue squares for goals that failed to plan
        failed_goals = logger.load_failed_plans()
        for _, x, _ in failed_goals:
            if contact_type == type:
                world.plot_block(ax, x, 'b')
        ax.set_title('%s Dataset' % logger.args.goal_type)
        ax.set_aspect('equal')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])

        fname = 'dataset_%s_%s.svg' % (ts, logger.args.goal_type)
        logger.save_figure(fname, dir=dir)
        plt.close()
