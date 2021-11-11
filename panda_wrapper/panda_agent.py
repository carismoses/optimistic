import numpy as np
import shutil
import os
import pybullet as p
import time

import pb_robot

class PandaAgent:
    def __init__(self, vis):
        """
        Build the Panda world in PyBullet using pb_robot. This matches the set up
        of George the Panda in the LIS lab
        """
        self.real = False # TODO: update to command line arg when use real robot
        robot_pose = np.eye(4)
        # Setup PyBullet instance to run in the background and handle planning/collision checking.
        self._planning_client_id = pb_robot.utils.connect(use_gui=False)
        self.plan()
        pb_robot.utils.set_default_camera()
        pb_robot.utils.set_camera(90, -15, 1.5)
        self.planning_robot = pb_robot.panda.Panda()
        self.planning_robot.arm.hand.Open()
        self.table = self.add_table()
        self.planning_robot.set_transform(np.eye(4))
        self.orig_joint_angles = self.planning_robot.arm.GetJointValues()


        # Setup PyBullet instance that only visualizes plan execution.
        # State needs to match the planning instance.
        self._execution_client_id = pb_robot.utils.connect(use_gui=vis)
        self.execute()
        pb_robot.utils.set_default_camera()
        pb_robot.utils.set_camera(0.1, -15, 1)
        self.execution_robot = pb_robot.panda.Panda()
        self.execution_robot.arm.hand.Open()
        self.execution_robot.set_transform(np.eye(4))
        self.add_table()

        self.remove_txt = []
        self.counter_id = None

        self.plan()


    def add_table(self):
        path_prefix = 'panda_wrapper/urdf_models'
        pb_robot_prefix = 'pb_robot/models'
        # add table
        table_x_offset = 0.2
        table_urdf = 'panda_table.urdf'
        table_point = [table_x_offset, 0, 0]
        shutil.copyfile(os.path.join(path_prefix, table_urdf),
                        os.path.join(pb_robot_prefix, table_urdf))
        file = os.path.join('models', table_urdf)
        table = pb_robot.body.createBody(file)
        table.set_point(table_point)
        return table


    def add_text(self, txt, position, size, color=(0,0,0), counter=False):
        self.execute()
        for id in self.remove_txt:
            pb_robot.viz.remove_debug(id)
        if counter and self.counter_id:
            pb_robot.viz.remove_debug(self.counter_id)
        txt_id = pb_robot.viz.add_text(txt, position=position, size=size, color=color)
        if counter:
            self.counter_id = txt_id
        else:
            self.remove_txt += [txt_id]
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
        self.planning_robot.arm.hand.Open()
        self.execute()
        self.execution_robot.arm.SetJointValues(self.orig_joint_angles)
        self.execution_robot.arm.hand.Open()
        print("Done")


    def get_init_state(self):
        """
        Get the PDDL representation of the robot and table.
        """
        conf = pb_robot.vobj.BodyConf(self.planning_robot, self.planning_robot.arm.GetJointValues())
        table_pose = pb_robot.vobj.BodyPose(self.table, self.table.get_base_link_pose())
        init = [('conf', conf),
                ('atconf', conf),
                ('handempty',),
                ('pose', self.table, table_pose),
                ('atpose', self.table, table_pose),
                ('table', self.table)]
        return init


    def execute_action(self, action, fixed, world_obstacles=[], pause=True, sim_fatal_failure_prob=0.0, sim_recoverable_failure_prob=0.0):
        obstacles = world_obstacles + fixed
        name, args = action
        executionItems = args[-1]
        self.execute()
        for e in executionItems:
            if self.real:
                e.simulate(timestep=0.1, obstacles=obstacles)
            else:
                if (name == 'move_contact') and isinstance(e, pb_robot.vobj.JointSpacePath):
                    e.simulate(timestep=0.5, obstacles=obstacles, control=True)
                else:
                    e.simulate(timestep=0.5, obstacles=obstacles)

            # Simulate failures if specified
            if (name in ["pick", "move_free"] and not isinstance(e, pb_robot.vobj.BodyGrasp)
                and not isinstance(e, pb_robot.vobj.MoveFromTouch)):
                if np.random.rand() < sim_fatal_failure_prob:
                    raise ExecutionFailure(fatal=True,
                        reason=f"Simulated fatal failure in {e}")
                elif np.random.rand() < sim_recoverable_failure_prob:
                    raise ExecutionFailure(fatal=False,
                        reason=f"Simulated recoverable failure in {e}")
            if pause:
                time.sleep(0.1)

        if self.real:
            input("Execute on Robot?")
            try:
                from franka_interface import ArmInterface
            except:
                print("Do not have rospy and franka_interface installed.")
                return
            arm = ArmInterface()
            arm.set_joint_position_speed(0.3)
            print("Executing on real robot")
            input("start?")
            for e in executionItems:
                e.execute(realRobot=arm, obstacles=obstacles)
        self.plan()
