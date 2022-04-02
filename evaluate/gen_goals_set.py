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

    xy_goals = [('yellow_block', (0.2, -0.3)),
                ('yellow_block', (0.1, -0.3)),
                ('yellow_block', (0.5, -0.3)),
                ('yellow_block', (0.6, -0.3)),
                ('yellow_block', (0.7, -0.3)),
                ('yellow_block', (0.2, -0.4)),
                ('yellow_block', (0.3, -0.4)),
                ('yellow_block', (0.2, -0.3)),
                ('yellow_block', (0.3, -0.2)),
                ('yellow_block', (0.4, 0.0))]

    goals = []
    for goal_obj, goal_xy in xy_goals:
        goal = world.generate_goal(goal_xy=goal_xy, goal_obj=goal_obj)
        goals.append(goal)
    for _ in range(10):
        goal = world.generate_goal(goal_obj='blue_block')
        goals.append(goal)

    with open(args.path, 'wb') as f: pickle.dump(goals, f)
