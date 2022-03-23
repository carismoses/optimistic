import argparse
import numpy as np

from domains.tools.world import ToolsWorld
from tamp.utils import execute_plan
from learning.datasets import OptDictDataset
from learning.utils import add_trajectory_to_dataset
from experiments.utils import ExperimentLogger
from experiments.skeletons import get_skeleton_fns, plan_from_skeleton


def gen_dataset(args):
    n_actions = 0

    # make logger and initialize dataset
    logger = ExperimentLogger.setup_experiment_directory(args, 'experiments')
    dataset = OptDictDataset()
    logger.save_trans_dataset(dataset, '', i=n_actions)

    # calculate the max dataset size
    max_len = 0
    for action, act_dict in dataset.datasets.items():
        for obj, act_obj_dict in act_dict.items():
            if (obj == 'blue_block' and action == 'pick') or \
                (obj == 'blue_block' and action == 'push-push_pull'):
                max_len += args.max_type_size
            else:
                max_len += 2*args.max_type_size

    # get all skeleton functions
    skeleton_fns = get_skeleton_fns()

    while len(dataset) < max_len:
        # randomly select a skeleton
        skel_i = np.random.randint(len(skeleton_fns))
        skeleton_fn = skeleton_fns[skel_i]

        # ground a plan from skeleton
        world = ToolsWorld()
        pddl_model_type = 'optimistic'
        goal_pred, add_to_state = world.generate_goal()
        skeleton = skeleton_fn(world, goal_pred)
        plan_info = plan_from_skeleton(skeleton, world, pddl_model_type, add_to_state)
        if plan_info is not None:
            pddl_plan, problem, init_expanded = plan_info

            # execute plan
            trajectory = execute_plan(world, problem, pddl_plan, init_expanded)

            if len(trajectory) > 0:
                # add to dataset
                add_trajectory_to_dataset('tools', logger, trajectory, world)
                n_actions += len(trajectory)

                # remove if unbalances dataset
                remove_point = False
                for action, act_dict in dataset.datasets.items():
                    for obj, act_obj_dict in act_dict.items():
                        if (obj == 'blue_block' and action == 'pick') or \
                            (obj == 'blue_block' and action == 'push-push_pull'):
                            if len(act_obj_dict) > args.max_type_size:
                                remove_point = True
                        elif len(act_obj_dict) > 2*args.max_type_size:
                            remove_point = True
                if remove_point:
                    logger.remove_dataset('', i=n_actions)
                    n_actions -= len(trajectory)
        dataset = logger.load_trans_dataset('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name',
                        type=str,
                        help='path to save datasets and models to (unless a restart, then use exp-path)')
    parser.add_argument('--max-type-size',
                        type=int,
                        help='max number of actions IN DATASET for each class in balanced case')
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    gen_dataset(args)
