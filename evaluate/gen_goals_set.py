import argparse
import dill as pickle

from domains.tools.world import ToolsWorld


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        type=str,
                        required=True,
                        help='path to save goals')
    parser.add_argument('--n-goals',
                        type=int,
                        required=True,
                        help='number of goals to generate')
    parser.add_argument('--debug',
                        action='store_true',
                        help='use to run in debug mode')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    world = ToolsWorld()

    goals = [world.generate_goal() for _ in range(args.n_goals)]

    with open(args.path, 'wb') as f: pickle.dump(goals, f)
