import os
import csv
from collections import defaultdict

# Map sub-folder to main category and sub-category
to_main_sub = {
    'fallen_trees': ('Public Safety', 'Fallen Trees'),
    'damaged_signs': ('Road Issues', 'Damaged Signs'),
    'potholes': ('Road Issues', 'Potholes'),
    'illegal_parking': ('Road Issues', 'Illegal Parking'),
    'graffiti': ('Public Cleanliness', 'Graffiti'),
    'garbage': ('Public Cleanliness', 'Garbage'),
}

base_dir = 'urban_issues_dataset'
splits = ['train', 'valid', 'test']

# Collect multi-labels for each image
image_labels = defaultdict(lambda: {'main_category': set(), 'sub_category': set(), 'split': None})

for subfolder, (main_cat, sub_cat) in to_main_sub.items():
    for split in splits:
        img_dir = os.path.join(base_dir, subfolder, split, 'images')
        if not os.path.exists(img_dir):
            continue
        for img_file in os.listdir(img_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                key = (img_file, split)
                image_labels[key]['main_category'].add(main_cat)
                image_labels[key]['sub_category'].add(sub_cat)
                image_labels[key]['split'] = split

# Write to CSV
with open('multi_label_dataset.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['image_filename', 'main_category', 'sub_category', 'split']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for (img_file, split), label_dict in image_labels.items():
        writer.writerow({
            'image_filename': img_file,
            'main_category': ';'.join(sorted(label_dict['main_category'])),
            'sub_category': ';'.join(sorted(label_dict['sub_category'])),
            'split': split
        })

print("Multi-label CSV created: multi_label_dataset.csv")
