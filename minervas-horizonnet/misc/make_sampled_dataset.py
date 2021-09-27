""" Random a dataset with given size """
import sys
import os
import shutil
import random
from tqdm import tqdm

if len(sys.argv) > 1:
    target_number = int(sys.argv[1])
    print(target_number)

src_folder = './Minervas_Layout_filtered_final'
list_fn = './Minervas_Layout_filtered_final/list.txt'
src_img_folder = os.path.join(src_folder, 'img')
src_label_folder = os.path.join(src_folder, 'label_cor')

with open(list_fn, 'r') as f:
    scene_list = f.read().splitlines()
    print('#', len(scene_list), 'samples in original dataset')

# sampled_list = random.choices(scene_list, k=target_number)
random.shuffle(scene_list)
sampled_list = scene_list[:target_number]

print(len(sampled_list))
# print(sampled_list)

target_folder = './Minervas_Layout_filtered_final_' + str(target_number)
os.makedirs(target_folder, exist_ok=True)
target_img_folder = os.path.join(target_folder, 'img')
target_label_folder = os.path.join(target_folder, 'label_cor')
os.makedirs(target_img_folder, exist_ok=True)
os.makedirs(target_label_folder, exist_ok=True)

for sampled_id in tqdm(sampled_list):
    sampled_id = sampled_id.split('.')[0]
    os.symlink(os.path.join('../../', src_img_folder, sampled_id+'.jpg'), os.path.join(target_img_folder, sampled_id+'.jpg'))
    os.symlink(os.path.join('../../', src_label_folder, sampled_id+'.txt'), os.path.join(target_label_folder, sampled_id+'.txt'))
    # break