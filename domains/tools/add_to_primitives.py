from learning.utils import model_forward

def get_trust_model(world, logger):
    def test(obj1, obj2, pose1, pose2, cont, fluents=[]):
        model = logger.load_trans_model(world)
        vec_state = world.state_to_vec(fluents)
        vec_action = world.pred_args_to_action_vec(obj1, obj2, pose1, pose2, cont)
        trust_model = model_forward(model, [*vec_state, vec_action], single_batch=True).mean().round().squeeze()
        return trust_model
    return test
