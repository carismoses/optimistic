from domains.tools.world import ToolsWorld
from domains.ordered_blocks.world import OrderedBlocksWorld

def init_world(domain, domain_args, pddl_model_type, vis, logger, planning_model_i=None):
    if domain == 'tools':
        return ToolsWorld.init(domain_args, pddl_model_type, vis, logger, planning_model_i=planning_model_i)
    elif domain == 'ordered_blocks':
        return OrderedBlocksWorld.init(domain_args, pddl_model_type, vis, logger)
    else:
        raise NotImplementedError
