from learning.datasets import model_forward

def get_trust_model(world, logger):
    def fn(top_block, bottom_block, fluents=[]):
        print('!!!! in here!')
        model = logger.load_trans_model()
        vec_state = world.state_to_vec(fluents)
        vec_action = world.action_to_vec(world.action_args_to_action(top_block, bottom_block))
        trust_model = model_forward(model, [*vec_state, vec_action]).round().squeeze()
        print('trust model:', trust_model)
        if trust_model:
            return (trust_model,)
        return ()
    return fn
