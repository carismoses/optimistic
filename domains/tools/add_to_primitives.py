from learning.utils import model_forward

def get_trust_model(world, logger, planning_model_i=None):
    def test(obj1, obj2, pose1, pose2, cont, fluents=[]):
        model = logger.load_trans_model(i=planning_model_i)
        x = world.pred_args_to_vec(obj1, obj2, pose1, pose2, cont)
        trust_model = model_forward(cont.type, model, x, single_batch=True).mean().round().squeeze()
        return trust_model
    return test
