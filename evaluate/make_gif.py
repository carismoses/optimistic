import imageio
import os

def make_gif(dir, files_path):
    images = []
    i = 0
    while True:
        filename = 'logs/experiments/%s/figures/goals/successes_%i.png' % (files_path, i)
        try:
            image = imageio.imread(filename)
        except:
            break
        images.append(image)
        i += 1
    gif_path = 'gifs/%s' % dir
    os.makedirs(gif_path, exist_ok=True)
    imageio.mimsave('%s/%s.gif' % (gif_path, files_path), images, duration=1)
    print(files_path)

dataset_paths = {#'random-actions':
                 #   ['random_actions-20220104-201317',
                 #   'random_actions-20220104-202422',
                 #   'random_actions-20220104-202440',
                 #   'random_actions-20220104-202447',
                 #   'random_actions-20220104-202453'],
                'random-goals-opt':
                    ['random_goals_opt-20220104-204547',
                    'random_goals_opt-20220104-203849',
                    'random_goals_opt-20220104-204627',
                    'random_goals_opt-20220104-204532',
                    'random_goals_opt-20220104-204536'],
                'sequential-goals':
                    ['sequential_goals-20220105-143004',
		    #'sequential_goals-20220105-143605',
		    'sequential_goals-20220105-143711',
	            'sequential_goals-20220105-145239',
                    'sequential_goals-20220105-145344'],
		'engineered-goals-dist':
		    ['engineered_goals_dist-20220112-162941',
		    'engineered_goals_dist-20220112-162947',
		    'engineered_goals_dist-20220112-162956',
		    'engineered_goals_dist-20220112-163004',
		    'engineered_goals_dist-20220112-163058'],
		'engineered-goals-size':
		    ['engineered_goals_size-20220112-172108',
		    'engineered_goals_size-20220112-172115',
		    'engineered_goals_size-20220112-172119',
		    'engineered_goals_size-20220112-172125',
 		    'engineered_goals_size-20220112-172129']
		}

for method, method_paths in dataset_paths.items():
    for path in method_paths:
        make_gif(method, path)
