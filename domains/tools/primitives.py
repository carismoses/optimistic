import numpy as np
from copy import copy
import random

import pb_robot
from pb_robot.tsrs.panda_box import ComputePrePose
from tsr.tsr import TSR

from tamp.utils import pause, Contact, vis_frame

DEBUG_FAILURE = False


def get_contact_gen(robot):
    def gen(obj1, obj2):
        # for now this only handles the case where obj1 is a tool and obj2 is a block
        block_dim = obj2.get_dimensions()[0] # block is a cuboid
        half_b = block_dim/2
        tool_thickness = 0.03
        half_tool = tool_thickness/2
        tool_length, tool_width = 0.4, 0.2+tool_thickness
        half_length, half_width = tool_length/2, tool_width/2

        rel_z = 0
        # for now defining 4 contact points
        rel_points_xy = [(-(half_length+half_b), 0),
                        ((half_length-tool_thickness-half_b), -(half_tool+half_b)),
                        ((half_length+half_b), -(half_width)),
                        ((half_length-half_tool), -(tool_width-half_tool+half_b))]

        contacts = []
        for rel_point_xy in rel_points_xy:
            rel_pose = ((*rel_point_xy, rel_z), (0., 0., 0., 1.))
            contact = Contact(obj1, obj2, rel_pose)
            contacts.append((contact,))
        random.shuffle(contacts)
        return contacts
    return gen


# try IK and collisions of EE contact pose
def ee_ik(ee_pose_world, robot, obstacles, seed_q=None):
    q = robot.arm.ComputeIK(ee_pose_world, seed_q=seed_q)
    if (q is None):
        if DEBUG_FAILURE: input('No Grasp IK: contact motion')
        return None
    if not robot.arm.IsCollisionFree(q, obstacles=obstacles, debug=DEBUG_FAILURE):
        if DEBUG_FAILURE: input('Grasp collision: contact motion')
        return None
    conf = pb_robot.vobj.BodyConf(robot, q)
    return conf


def get_contact_motion_gen(robot, fixed=[], num_attempts=20):
    # obj1 is tool in grasp, obj2 is at pose1, cont is in obj2 frame
    def fn(obj1, grasp, obj2, pose1, pose2, cont):
        # ee pose at contact
        obj2_world = pb_robot.geometry.tform_from_pose(pose1.pose)
        cont_tform = pb_robot.geometry.tform_from_pose(cont.rel_pose)
        obj1_contact_world = obj2_world@cont_tform
        ee_contact_world = obj1_contact_world@grasp.grasp_objF

        # ee pose at beginning of approach
        approach_dist = 0.1
        dir = pose1.pose[0]-pose2.pose[0]
        unit_dir = dir/np.linalg.norm(dir)
        approach = np.eye(4)
        approach[:3,3] = approach_dist*unit_dir
        obj1_approach_world = obj2_world@approach@cont_tform
        ee_approach_world = obj1_approach_world@grasp.grasp_objF

        # ee pose at end of push path
        obj1_pose2_world = pb_robot.geometry.tform_from_pose(pose2.pose)@\
                            pb_robot.geometry.tform_from_pose(cont.rel_pose)
        ee_pose2_world = obj1_pose2_world@grasp.grasp_objF

        for ax in range(num_attempts):
            ## debugging
            #tforms = [pb_robot.geometry.tform_from_pose(pose.pose),
            #            obj1_contact_world,
            #            ee_contact_world,
            #            obj1_approach_world,
            #            ee_approach_world]
            #for ti, tform in enumerate(tforms):
            #    vis_frame(pb_robot.geometry.pose_from_tform(tform), 0)
            #    if ti in [1, 3]:
            #        obj1.set_base_link_pose(pb_robot.geometry.pose_from_tform(tform))
            #    pause()
            ##

            # calculate all configurations from poses
            obstacles = copy(fixed)
            conf_contact = ee_ik(ee_contact_world, robot, obstacles)
            if not conf_contact: continue
            obstacles = copy(fixed) + [obj2]
            conf_approach = ee_ik(ee_approach_world, robot, obstacles, seed_q=conf_contact.configuration)
            if not conf_approach: continue
            obstacles = copy(fixed)
            conf_pose2 = ee_ik(ee_pose2_world, robot, obstacles)
            if not conf_pose2: continue

            # contact motion
            # TODO: constrain path to be straight line (I think I can add constraints to birrt call
            # but unsure the proper way to do so, so for now interpolate between
            # ee positions assuming orientation is constant)
            #push_path = robot.arm.birrt.PlanToConfiguration(robot.arm,
            #                                                conf_contact.configuration,
            #                                                conf_pose2.configuration,
            #                                                obstacles=obstacles)
            #if push_path is None:
            #    if DEBUG_FAILURE: input('Push motion failed')
            #    continue
            path_len = 20
            push_ee_positions = np.linspace(ee_contact_world[:3,3],
                                    ee_pose2_world[:3,3],
                                    path_len)
            push_ee_orn = ee_contact_world[1]
            push_path = [conf_contact.configuration]
            seed_q = conf_contact.configuration
            for ee_pos in push_ee_positions[1:-1]:
                ee_tform = copy(ee_contact_world)
                ee_tform[:3,3] = ee_pos
                for _ in range(num_attempts):
                    push_conf = ee_ik(ee_tform, robot, obstacles, seed_q=seed_q)
                    if push_conf:
                        break
                if not push_conf:
                    return None
                push_path.append(push_conf.configuration)
                seed_q = push_conf.configuration
            push_path.append(conf_pose2.configuration)
            push_path = np.array(push_path)


            # TODO: add a path that breaks contact from the object
            command = [pb_robot.vobj.MoveToTouch(robot.arm,
                                                    conf_approach.configuration,
                                                    conf_contact.configuration,
                                                    None,
                                                    obj2),
                        pb_robot.vobj.JointSpacePushPath(robot.arm,
                                                    push_path)]
            return (conf_approach, conf_pose2, command)
        return None
    return fn


def get_free_motion_gen(robot, fixed=[]):
    def fn(conf1, conf2, fluents=[]):
        obstacles = assign_fluent_state(fluents)
        fluent_names = [o.get_name() for o in obstacles]
        for o in fixed:
            if o.get_name() not in fluent_names:
                obstacles.append(o)

        path = robot.arm.birrt.PlanToConfiguration(robot.arm, conf1.configuration, conf2.configuration, obstacles=obstacles)

        if path is None:
            if DEBUG_FAILURE: input('Free motion failed')
            return None
        command = [pb_robot.vobj.JointSpacePath(robot.arm, path)]
        return (command,)
    return fn


def get_holding_motion_gen(robot, fixed=[]):
    def fn(conf1, conf2, obj, grasp, fluents=[]):
        obstacles = assign_fluent_state(fluents)
        fluent_names = [o.get_name() for o in obstacles]
        for o in fixed:
            if o.get_name() not in fluent_names:
                obstacles.append(o)

        old_q = robot.arm.GetJointValues()
        orig_pose = obj.get_base_link_pose()
        robot.arm.SetJointValues(conf1.configuration)
        robot.arm.Grab(obj, grasp.grasp_objF)

        path = robot.arm.birrt.PlanToConfiguration(robot.arm, conf1.configuration, conf2.configuration, obstacles=obstacles)

        robot.arm.Release(obj)
        obj.set_base_link_pose(orig_pose)
        robot.arm.SetJointValues(old_q)

        if path is None:
            if DEBUG_FAILURE: input('Holding motion failed')
            return None
        command = [pb_robot.vobj.JointSpacePath(robot.arm, path)]
        return (command,)
    return fn


def get_ik_fn(robot, fixed=[], num_attempts=4, approach_frame='gripper', backoff_frame='global'):
    def fn(obj, pose, grasp, return_grasp_q=False, check_robust=False):
        obstacles = copy(fixed) # for some reason  adding to fixed here changes it in other primitives, so use copy of fixed
        if approach_frame == 'global': # grasp object for collision checking on place action
            orig_pose = obj.get_base_link_pose()
            robot.arm.Grab(obj, grasp.grasp_objF)
        else:                           # avoid object to be picked when pick action
            obstacles += [obj]
        obj_worldF = pb_robot.geometry.tform_from_pose(pose.pose)
        grasp_worldF = np.dot(obj_worldF, grasp.grasp_objF)
        grasp_worldR = grasp_worldF[:3,:3]

        e_x, e_y, e_z = np.eye(3) # basis vectors

        # The x-axis of the gripper points toward the camera
        # The y-axis of the gripper points along the plane of the hand
        # The z-axis of the gripper points forward

        is_top_grasp = grasp_worldR[:,2].dot(-e_z) > 0.999
        is_upside_down_grasp = grasp_worldR[:,2].dot(e_z) > 0.001
        is_gripper_sideways = np.abs(grasp_worldR[:,1].dot(e_z)) > 0.999
        is_camera_down = grasp_worldR[:,0].dot(-e_z) > 0.999
        is_wrist_too_low = grasp_worldF[2,3] < 0.088/2 + 0.005


        if is_gripper_sideways:
            return None
        if is_upside_down_grasp:
            return None
        if is_camera_down:# and approach_frame == 'gripper':
            return None

        # the gripper is too close to the ground. the wrist of the arm is 88mm
        # in diameter, and it is the widest part of the hand. Include a 5mm
        # clearance
        if not is_top_grasp and is_wrist_too_low:
            return None
        # If the obj/gripper is in the storage area, don't use low grasps.
        if grasp_worldF[0,3] < 0.2 and grasp_worldF[2,3] < 0.1:
            return None

        if approach_frame == 'gripper':
            approach_tform = ComputePrePose(grasp_worldF, [0, 0, -0.1], approach_frame)
        elif approach_frame == 'global':
            approach_tform = ComputePrePose(grasp_worldF, [0, 0, 0.1], approach_frame) # Was -0.125
        else:
            raise NotImplementedError()

        if backoff_frame == 'gripper':
            backoff_tform = ComputePrePose(grasp_worldF, [0, 0, -0.1], backoff_frame)
        elif backoff_frame == 'global':
            backoff_tform = ComputePrePose(grasp_worldF, [0, 0, 0.1], backoff_frame) # Was -0.125
        else:
            raise NotImplementedError()

        for ax in range(num_attempts):
            q_grasp = robot.arm.ComputeIK(grasp_worldF)
            if (q_grasp is None):
                if DEBUG_FAILURE: input('No Grasp IK: pick/place')
                continue
            if not robot.arm.IsCollisionFree(q_grasp, obstacles=obstacles, debug=DEBUG_FAILURE):
                if DEBUG_FAILURE: input('Grasp collision: pick/place')
                continue

            q_approach = robot.arm.ComputeIK(approach_tform, seed_q=q_grasp)
            if (q_approach is None):
                if DEBUG_FAILURE: input('No approach IK')
                continue
            if not robot.arm.IsCollisionFree(q_approach, obstacles=obstacles, debug=DEBUG_FAILURE):
                if DEBUG_FAILURE: input('Approach motion collision')
                continue
            conf_approach = pb_robot.vobj.BodyConf(robot, q_approach)

            # Only recompute the backoff if it's different from the approach.
            if approach_frame == backoff_frame:
                q_backoff = q_approach
            else:
                q_backoff = robot.arm.ComputeIK(backoff_tform, seed_q=q_grasp)
                if (q_backoff is None):
                    if DEBUG_FAILURE: input('No backoff IK')
                    continue
                if not robot.arm.IsCollisionFree(q_backoff, obstacles=obstacles, debug=DEBUG_FAILURE):
                    if DEBUG_FAILURE: input('Backoff motion collision')
                    continue
            conf_backoff = pb_robot.vobj.BodyConf(robot, q_backoff)

            path_approach = robot.arm.snap.PlanToConfiguration(robot.arm, q_approach, q_grasp, obstacles=obstacles)
            if path_approach is None:
                if DEBUG_FAILURE: input('Approach motion failed')
                continue
            if backoff_frame == 'global':
                path_backoff = robot.arm.snap.PlanToConfiguration(robot.arm, q_grasp, q_backoff, obstacles=obstacles, check_upwards=True)
            else:
                path_backoff = robot.arm.snap.PlanToConfiguration(robot.arm, q_grasp, q_backoff, obstacles=obstacles, check_upwards=False)
            if path_backoff is None:
                if DEBUG_FAILURE: input('Backoff motion failed')
                continue

            # If the grasp is valid, check that it is robust (i.e., also valid under pose estimation error).
            if check_robust:
                for _ in range(10):
                    x, y, z = pose.pose[0]
                    new_pose = ((x + np.random.randn()*0.02, y + np.random.randn()*0.02, z), pose.pose[1])
                    new_pose = pb_robot.vobj.BodyPose(obj, new_pose)
                    valid = fn(obj, pose, grasp, check_robust=False)
                    if not valid:
                        print('Grasp not robust')
                        print(x - new_pose.pose[0][0], y - new_pose.pose[0][1])
                        return None

            command = [pb_robot.vobj.MoveToTouch(robot.arm, q_approach, q_grasp, grasp, obj),
                       grasp,
                       pb_robot.vobj.MoveFromTouch(robot.arm, q_backoff)]
            if approach_frame == 'global':
                robot.arm.Release(obj)
                obj.set_base_link_pose(orig_pose)

            if return_grasp_q:
                return (pb_robot.vobj.BodyConf(robot, q_grasp),)
            return (conf_approach, conf_backoff, command)
        return None
    return fn


# placement pose for a block
def get_pose_gen_block(fixed=[]):
    def fn(top_block, bottom_block, bottom_block_pose):
        # NOTE: this assumes we want all blocks at the same orientation always (when not held)
        # and that all blocks start off at orn (0,0,0,1)
        bottom_block_tform = pb_robot.geometry.tform_from_pose(bottom_block_pose.pose)
        rel_z_pose = bottom_block.get_dimensions()[2]/2+top_block.get_dimensions()[2]/2
        rel_tform = np.array([[1.  , 0.  , 0.  , 0.  ],
                            [0.  , 1.  , 0.  , 0.  ],
                            [0.  , 0.  , 1.  , rel_z_pose],
                            [0.  , 0.  , 0.  , 1.  ]])
        top_block_tform = bottom_block_tform@rel_tform
        top_block_pose = pb_robot.geometry.pose_from_tform(top_block_tform)
        top_block_pose  = pb_robot.vobj.BodyPose(top_block, top_block_pose)
        return (top_block_pose,)
    return fn


def get_block_grasp_gen(robot, add_slanted_grasps=True, add_orthogonal_grasps=True):
    # add_slanted_grasps = True
    # I opt to use TSR to define grasp sets but you could replace this
    # with your favorite grasp generator
    def gen(block):
        grasp_tsr = pb_robot.tsrs.panda_box.grasp(block,
            add_slanted_grasps=add_slanted_grasps, add_orthogonal_grasps=add_orthogonal_grasps)
        grasps = []

        for sampled_tsr in grasp_tsr:
            grasp_worldF = sampled_tsr.sample()
            grasp_objF = np.dot(np.linalg.inv(block.get_base_link_transform()), grasp_worldF)
            block_grasp = pb_robot.vobj.BodyGrasp(block, grasp_objF, robot.arm)
            grasps.append((block_grasp,))
        return grasps
    return gen


def get_tool_grasp_gen(robot, add_slanted_grasps=True, add_orthogonal_grasps=True):
    def gen(tool):
        tool_thickness = 0.03
        ee_to_palm_distance = 0.1034
        z_offset = ee_to_palm_distance + tool_thickness/2
        p0_w = tool.get_base_link_pose()
        T0_w = pb_robot.geometry.tform_from_pose(p0_w)

        # top down grasp transforms (grasp face perpendicular to z-axis)
        Tw_e_side1 = np.array([[ 1., 0.,  0., 0.0],
                                [ 0.,-1.,  0., 0.0],
                                [ 0., 0., -1., z_offset], # Added tmp.
                                [ 0., 0.,  0., 1.]])

        Tw_e_side2 = np.array([[ 1., 0., 0., 0.0],
                                [ 0., 1., 0., 0.0],
                                [ 0., 0., 1., -z_offset],
                                [ 0., 0., 0., 1.]])

        Tw_e_side3 = np.array([[ 0., 1.,  0., 0.0],
                                [ 1., 0.,  0., 0.0],
                                [ 0., 0., -1., z_offset],
                                [ 0., 0.,  0., 1.]])

        Tw_e_side4 = np.array([[ 0., 1., 0., 0.0],
                                [-1., 0., 0., 0.0],
                                [ 0., 0., 1., -z_offset],
                                [ 0., 0., 0., 1.]])

        # hook grasps
        hook_offset_xy = (0.1, 0.0)

        # poke grasps
        poke_offset_xy = (-.185, 0.115)

        Bw = np.zeros((6,2))
        grasp_tsrs = []
        for Tw_e in [Tw_e_side1, Tw_e_side2, Tw_e_side3, Tw_e_side4]:
            for x_offset, y_offset in [hook_offset_xy, poke_offset_xy]:
                Tw_e_adjust = copy(Tw_e)
                Tw_e_adjust[0][3] += x_offset
                Tw_e_adjust[1][3] += y_offset
                grasp_tsrs.append(TSR(T0_w = T0_w, Tw_e = Tw_e_adjust, Bw = Bw))

        grasps = []
        for tsr in grasp_tsrs:
            grasp_worldF = tsr.sample()
            grasp_objF = np.dot(np.linalg.inv(tool.get_base_link_transform()), grasp_worldF)
            tool_grasp = pb_robot.vobj.BodyGrasp(tool, grasp_objF, robot.arm)
            ## for debugging
            #try:
            #    q_grasp = robot.arm.ComputeIK(grasp_worldF)
            #    conf = pb_robot.vobj.BodyConf(robot, q_grasp)
            #    robot.arm.SetJointValues(conf.configuration)
            #    pause()
            #except:
            #    input('impossible grasp IK')
            ##
            grasps.append((tool_grasp,))
        random.shuffle(grasps)
        return grasps
    return gen


def assign_fluent_state(fluents):
    obstacles = []
    for fluent in fluents:
        name, args = fluent[0], fluent[1:]
        if name == 'atpose':
            o, p = args
            obstacles.append(o)
            o.set_base_link_pose(p.pose)
        else:
            raise ValueError(name)
    return obstacles
