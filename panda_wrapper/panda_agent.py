import numpy as np
import shutil
import os
import pybullet as p
import time

import pb_robot

class PandaAgent:
    def __init__(self):
        """
        Build the Panda world in PyBullet using pb_robot. This matches the set up
        of George the Panda in the LIS lab
        """
        robot_pose = np.eye(4)
        # Setup PyBullet instance to run in the background and handle planning/collision checking.
        self._planning_client_id = pb_robot.utils.connect(use_gui=False)
        self.plan()
        pb_robot.utils.set_default_camera()
        self.planning_robot = pb_robot.panda.Panda()
        self.planning_robot.arm.hand.Open()
        self.planning_robot.set_transform(robot_pose)
        self.orig_joint_angles = self.planning_robot.arm.GetJointValues()
        self.fixed = self.add_fixed_objects()

        # Setup PyBullet instance that only visualizes plan execution.
        # State needs to match the planning instance.
        self._execution_client_id = pb_robot.utils.connect(use_gui=True)
        self.execute()
        pb_robot.utils.set_default_camera()
        self.execution_robot = pb_robot.panda.Panda()
        self.execution_robot.arm.hand.Open()
        self.execution_robot.set_transform(robot_pose)
        self.add_fixed_objects()

        self.plan()


    def add_fixed_objects(self):
        fixed_objects = []
        # add table, frame, and collision walls to model
        table_x_offset = 0.2
        urdfs_and_points = [('panda_table.urdf', [table_x_offset,
                                                    0,
                                                    0]),
                            ('panda_frame.urdf', [table_x_offset + 0.762 - 0.0127,
                                                    0 + 0.6096 - 0.0127,
                                                    0]),
                            ('walls.urdf', [table_x_offset + 0.762 + 0.005,
                                                    0,
                                                    0])]
        path_prefix = 'panda_wrapper/urdf_models'
        pb_robot_prefix = 'pb_robot/models'
        for urdf, point in urdfs_and_points:
            shutil.copyfile(os.path.join(path_prefix, urdf),
                            os.path.join(pb_robot_prefix, urdf))
            file = os.path.join('models', urdf)
            pb_object = pb_robot.body.createBody(file)
            pb_object.set_point(point)
            fixed_objects.append(pb_object)
        # HACK
        self.table = fixed_objects[0]
        return fixed_objects


    def _add_text(self, txt):
        self.execute()
        pb_robot.viz.remove_all_debug()
        self.txt_id = pb_robot.viz.add_text(txt, position=(0, 0.25, 0.75), size=2)
        self.plan()


    def execute(self):
        self.state = 'execute'
        pb_robot.aabb.set_client(self._execution_client_id)
        pb_robot.body.set_client(self._execution_client_id)
        pb_robot.collisions.set_client(self._execution_client_id)
        pb_robot.geometry.set_client(self._execution_client_id)
        pb_robot.grasp.set_client(self._execution_client_id)
        pb_robot.joint.set_client(self._execution_client_id)
        pb_robot.link.set_client(self._execution_client_id)
        pb_robot.panda.set_client(self._execution_client_id)
        pb_robot.planning.set_client(self._execution_client_id)
        pb_robot.utils.set_client(self._execution_client_id)
        pb_robot.viz.set_client(self._execution_client_id)


    def plan(self):
        self.state = 'plan'
        pb_robot.aabb.set_client(self._planning_client_id)
        pb_robot.body.set_client(self._planning_client_id)
        pb_robot.collisions.set_client(self._planning_client_id)
        pb_robot.geometry.set_client(self._planning_client_id)
        pb_robot.grasp.set_client(self._planning_client_id)
        pb_robot.joint.set_client(self._planning_client_id)
        pb_robot.link.set_client(self._planning_client_id)
        pb_robot.panda.set_client(self._planning_client_id)
        pb_robot.planning.set_client(self._planning_client_id)
        pb_robot.utils.set_client(self._planning_client_id)
        pb_robot.viz.set_client(self._planning_client_id)


    def reset(self):
        """ Resets the planning world to its original configuration """
        print("Resetting world")
        self.plan()
        self.planning_robot.arm.SetJointValues(self.orig_joint_angles)
        self.execute()
        self.execution_robot.arm.SetJointValues(self.orig_joint_angles)
        print("Done")


    def get_init_state(self):
        """
        Get the PDDL representation of the robot and table.
        """
        conf = pb_robot.vobj.BodyConf(self.planning_robot, self.planning_robot.arm.GetJointValues())
        print('Initial configuration:', conf.configuration)
        init = [('canmove',),
                ('conf', conf),
                ('startconf', conf),
                ('atconf', conf),
                ('handempty',)]

        self.table_pose = pb_robot.vobj.BodyPose(self.table, self.table.get_base_link_pose())
        init += [('pose', self.table, self.table_pose),
                 ('atpose', self.table, self.table_pose)]
        return init

    def step_simulation(self):
        print('Press Ctrl-C to resume execution.')
        try:
            while True:
                p.stepSimulation(physicsClientId=self._execution_client_id)
                p.stepSimulation(physicsClientId=self._planning_client_id)
                time.sleep(1/2400.)
        except KeyboardInterrupt:
            pass
