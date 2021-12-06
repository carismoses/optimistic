from domains.tools.world import ToolsWorld
from domains.ordered_blocks.world import OrderedBlocksWorld

def init_world(domain, domain_args, pddl_model_type, vis, logger):
    if domain == 'tools':
        return ToolsWorld.init(domain_args, pddl_model_type, vis, logger)
    elif domain == 'ordered_blocks':
        return OrderedBlocksWorld.init(domain_args, pddl_model_type, vis, logger)
    else:
        raise NotImplementedError