from learning.utils import model_forward

def get_trust_model(world, logger):
    def test(top_block, bottom_block, fluents=[]):
        model = logger.load_model('trans')
        vec_state = world.state_to_vec(fluents)
        vec_action = world.action_to_vec(world.action_args_to_action(top_block, bottom_block))
        trust_model = model_forward(model, [*vec_state, vec_action]).round().squeeze()
        return trust_model
    return test
