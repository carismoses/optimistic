"""
Utilities for ordered blocks planning with PDDLStream
"""
from pddlstream.utils import read

def get_pddlstream_info(world):
    """ Gets information for PDDLStream planning problem """
    domain_pddl = read('domains/ordered_blocks/domain.pddl')
    stream_pddl = None
    constant_map = {}
    stream_map = {}
    return domain_pddl, constant_map, stream_pddl, stream_map
