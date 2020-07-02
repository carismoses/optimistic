import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt

from actions import plan_action
from agents.teleport_agent import TeleportAgent
from block_utils import Object, Dimensions, Position, Color
from particle_belief import ParticleBelief
from tower_planner import TowerPlanner


def get_adversarial_blocks():
    b1 = Object(name='block1',
                dimensions=Dimensions(0.02, 0.06, 0.02),
                mass=1.,
                com=Position(0.045, 0.145, 0),
                color=Color(0, 0, 1))
    b2 = Object(name='block2',
                dimensions=Dimensions(0.01, 0.06, 0.02),
                mass=1.,
                com=Position(-0.024, -0.145, 0),
                color=Color(1, 0, 1))
    b3 = Object(name='block3',
                dimensions=Dimensions(0.01, 0.01, 0.1),
                mass=1.,
                com=Position(0, 0, 0.2),
                color=Color(0, 1, 1))
    b4 = Object(name='block4',
                dimensions=Dimensions(0.06, 0.01, 0.02),
                mass=1.,
                com=Position(-0.145, 0, -0.03),
                color=Color(0, 1, 0))
    return [b1, b2, b3, b4]


def plot_com_error(errors_random, errors_var):

    for tx in range(0, len(errors_var[0][0])):
        err_rand, err_var = 0, 0
        for bx in range(0, len(errors_var)):
            true = np.array(errors_var[bx][1])
            guess_rand = errors_random[bx][0][tx]
            guess_var = errors_var[bx][0][tx]
            err_var += np.linalg.norm(true-guess_var)
            err_rand += np.linalg.norm(true-guess_rand)
        plt.scatter(tx, err_rand/len(errors_var), c='r')
        plt.scatter(tx, err_var/len(errors_var), c='b')
    plt.show()    


def main(args):
    # get a bunch of random blocks
    blocks = get_adversarial_blocks()

    if args.agent == 'teleport':
        agent = TeleportAgent()
    else:
        raise NotImplementedError()

    # construct a world containing those blocks
    for block in blocks:
        # new code
        print('Running filter for', block.name)
        belief = ParticleBelief(block, N=10, plot=args.plot, vis_sim=args.vis)
        for interaction_num in range(2):
            print("Interaction number: ", interaction_num)
            action = plan_action(belief, exp_type='random', action_type='place')
            observation = agent.simulate_action(action, block)
            belief.update(observation)
            block.com_filter = belief.particles

    # find the tallest tower
    print('Finding tallest tower.')
    tp = TowerPlanner()
    tallest_tower = tp.plan(blocks)

    # and visualize the result
    agent.simulate_tower(tallest_tower, vis=True, T=100, save_tower=args.save_tower)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--save-tower', action='store_true')
    parser.add_argument('--agent', choices=['teleport', 'panda'], default='teleport')
    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    # test_exploration(args)

    main(args)
