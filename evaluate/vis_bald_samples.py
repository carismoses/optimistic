import matplotlib.pyplot as plt
from domains.tools.world import ToolsWorld
from domains.tools.primitives import get_contact_gen

from experiments.utils import ExperimentLogger


goals_path = 'logs/experiments/bald-goals-20220302-095132'
goals_logger = ExperimentLogger(goals_path)
goals = goals_logger.load_goals()

world = ToolsWorld(False, None, ['poke', 'push_pull'])
contacts_fn = get_contact_gen(world.panda.planning_robot, world.contact_types)
contacts = contacts_fn(world.objects['tool'], world.objects['yellow_block'], shuffle=False)

fig, axes = plt.subplots(1, figsize=(5, 5))

colors = {'poke':'m', 'push_pull':'c'}
count = {'poke': 0, 'push_pull': 0}
for type in world.contact_types:
    for final_pose, contact in goals:
        if contact.type == type:
            count[type] += 1
            point = final_pose.pose[0][:2]
            axes.plot(*point, colors[type]+'.')

    for contactw in contacts:
        cont = contactw[0]
        if cont.type == type:
            world.vis_tool_ax(cont, axes, frame='world', color=colors[type])

    y0 = (0.4, -0.3)
    xs = [y0[0]-.5, y0[0]+.6]
    ys = [y0[1]-.25, y0[1]+.45]
    axes.set_xlim(xs)
    axes.set_ylim(ys)
    #axes[1].set_xlim(xs)
    #axes[1].set_ylim(ys)

world.plot_block(axes, world.init_objs_pos_xy['yellow_block'], color='k')
print(count)
plt.show()
