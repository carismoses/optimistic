import re
import imageio
import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

def make_gif(fig_dir, files_path, file_match_str, svg=True):
    #import pdb; pdb.set_trace()
    images = {0: [], 1: [], 2: [], 3:[]}
    images_path = os.path.join(files_path, 'figures', fig_dir)
    image_file_names = os.listdir(images_path)

    # iterate through adding to appropriate list
    for image_file_name in sorted(image_file_names):
        png_file_name = image_file_name[:-3] + '.jpg'
        matches = re.match(match_str, image_file_name)
        if matches:
            contact_i = int(matches.group(3))
            if svg:
                drawing = svg2rlg(os.path.join(images_path, image_file_name))
                renderPM.drawToFile(drawing, os.path.join(images_path, png_file_name), fmt="PNG")
                image = imageio.imread(os.path.join(images_path, png_file_name))
            else:
                image = imageio.imread(os.path.join(images_path, image_file_name))
            images[contact_i].append(image)

    gif_path = os.path.join(files_path, 'figures', 'gifs')
    os.makedirs(gif_path, exist_ok=True)
    for cont_i, images in images.items():
        imageio.mimsave(os.path.join(gif_path, 'cont_%i.gif'%cont_i), images, duration=.3)

exp_path = 'logs/experiments/sequential-goals-20220208-025351'
            #'logs/experiments/sequential-goals-20220208-025356'
fig_dir = 'accuracy'
match_str = r'acc_(.*)-(.*)_(.*).svg'
make_gif(fig_dir, exp_path, match_str)
