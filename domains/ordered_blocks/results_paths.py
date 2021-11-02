# 4-block results
random_actions = ['logs/experiments/random-actions-20210909-101948',
                    'logs/experiments/random-actions-20210909-102017',
                    'logs/experiments/random-actions-20210909-102039',
                    'logs/experiments/random-actions-20210909-102106',
                    'logs/experiments/random-actions-20210909-102133']

random_goals_opt = ['logs/experiments/random-goals-opt-20210909-102114',
                    'logs/experiments/random-goals-opt-20210909-102431',
                    'logs/experiments/random-goals-opt-20210909-102746',
                    'logs/experiments/random-goals-opt-20210909-103106',
                    'logs/experiments/random-goals-opt-20210909-103421']

long_horizon_goals_opt = ['logs/experiments/long-goals-random-goals-opt-20210913-160132',
                            'logs/experiments/long-goals-random-goals-opt-20210913-161215',
                            'logs/experiments/long-goals-random-goals-opt-20210913-162300',
                            'logs/experiments/long-goals-random-goals-opt-20210913-163341',
                            'logs/experiments/long-goals-random-goals-opt-20210913-164433']

random_goals_learned = ['logs/experiments/random-goals-learned-20210913-115712',
                        'logs/experiments/random-goals-learned-20210913-120030',
                        'logs/experiments/random-goals-learned-20210913-120350',
                        'logs/experiments/random-goals-learned-20210913-120711',
                        'logs/experiments/random-goals-learned-20210913-121032']

long_horizon_goals_learned = ['logs/experiments/long-goals-random-goals-learned-20210914-104404',
                                'logs/experiments/long-goals-random-goals-learned-20210914-104532',
                                'logs/experiments/long-goals-random-goals-learned-20210914-104710',
                                'logs/experiments/long-goals-random-goals-learned-20210914-104852',
                                'logs/experiments/long-goals-random-goals-learned-20210914-105019']

#model_paths = {'random-actions': random_actions,
                #'random-goals-opt': random_goals_opt,
                #'random-goals-learned': random_goals_learned,
                #'long-horizon-goals-opt': long_horizon_goals_opt,
#                'long-horizon-goals-learned': long_horizon_goals_learned}

# 6 block results (all after explore_random() bug fix)
curriculum_goals_learned_6 = ['logs/experiments/curriculum-goals-learned-6-fixed-20210916-163130',
                                'logs/experiments/curriculum-goals-learned-6-fixed-20210916-164410',
                                'logs/experiments/curriculum-goals-learned-6-fixed-20210916-165512',
                                'logs/experiments/curriculum-goals-learned-6-fixed-20210916-170835',
                                'logs/experiments/curriculum-goals-learned-6-fixed-20210916-173651']

random_goals_learned_6 = ['logs/experiments/random-goals-learned-6-fixed-20210916-142426',
                            'logs/experiments/random-goals-learned-6-fixed-20210916-143553',
                            'logs/experiments/random-goals-learned-6-fixed-20210916-144821',
                            'logs/experiments/random-goals-learned-6-fixed-20210916-163129',
                            'logs/experiments/random-goals-learned-6-fixed-20210916-164612']

# these results are post bug-fix
random_actions_6 = ['logs/experiments/random-actions-6-blocks-20210916-120424',
                    'logs/experiments/random-actions-6-blocks-20210916-122625',
                    'logs/experiments/random-actions-6-blocks-20210916-124801',
                    'logs/experiments/random-actions-6-blocks-20210916-101032',
                    'logs/experiments/random-actions-6-blocks-20210916-103213']

## New curriculum methods with 6 blocks
new_curric_25 = ['logs/experiments/curric_new_25-20210920-125116',
                    'logs/experiments/curric_new_25-20210920-130301',
                    'logs/experiments/curric_new_25-20210920-131525',
                    'logs/experiments/curric_new_25-20210920-143357',
                    'logs/experiments/curric_new_25-20210920-145528']

new_new_curric_25 = ['logs/experiments/curric_new_25_not_random_curric_goal-20210920-143404',
                    'logs/experiments/curric_new_25_not_random_curric_goal-20210920-144527',
                    'logs/experiments/curric_new_25_not_random_curric_goal-20210920-145813',
                    'logs/experiments/curric_new_25_not_random_curric_goal-20210920-150904',
                    'logs/experiments/curric_new_25_not_random_curric_goal-20210920-151747']

'''
model_paths = {#'random-goals-learned': random_goals_learned_6,
                #'random-actions': random_actions_6,
                'curriculum-goals-learned': curriculum_goals_learned_6,
                'new-curric': new_curric_25,
                'new-new-curric': new_new_curric_25}
'''

# 4 blocks (fixed so opt planning generates different plans for same goal)
fixed_opt = ['logs/experiments/random-goals-learned-fixed-20211101-113354',
                'logs/experiments/random-goals-opt-20211101-142226',
                'logs/experiments/random-goals-opt-20211101-144408',
                'logs/experiments/random-goals-opt-20211101-150659',
                'logs/experiments/random-goals-opt-20211101-152928']

model_paths = {'random-goals-learned': long_horizon_goals_learned,
                'random-goals-opt': fixed_opt,
                'random-actions': random_actions}
