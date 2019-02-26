import json
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

seed = 999
random.seed(seed)

src_dir = Path('../phase1/').resolve()
dst_dir = Path('./data/train/').resolve()
if dst_dir.exists():
    shutil.rmtree(str(dst_dir))
dst_dir.mkdir(parents=True)

with (src_dir / 'train_list.txt').open() as f:
    data = [line.split() for line in f.readlines()]

anns = []
subjects = set()
for i, (rgb_path, dep_path, ifr_path, lbl) in enumerate(tqdm(data)):
    src_rgb_path = src_dir / rgb_path
    src_dep_path = src_dir / dep_path
    src_ifr_path = src_dir / ifr_path
    dst_rgb_path = dst_dir / f'{i:08d}.rgb.jpg'
    dst_dep_path = dst_dir / f'{i:08d}.dep.jpg'
    dst_ifr_path = dst_dir / f'{i:08d}.ifr.jpg'

    Image.open(src_rgb_path).resize((64, 64)).save(dst_rgb_path)
    Image.open(src_dep_path).resize((64, 64)).save(dst_dep_path)
    Image.open(src_ifr_path).resize((64, 64)).save(dst_ifr_path)

    subject = src_rgb_path.parents[2].stem[-4:]
    subjects.add(subject)

    anns.append({
        'subject': subject,
        'lbl': int(lbl),
        'rgb_path': str(dst_rgb_path),
        'dep_path': str(dst_dep_path),
        'ifr_path': str(dst_ifr_path),
        'raw_rgb_path': str(src_rgb_path),
        'raw_dep_path': str(src_dep_path),
        'raw_ifr_path': str(src_ifr_path),
    })
    
subjects = list(subjects)
random.shuffle(subjects)
train_subjects = subjects[:len(subjects) * 4 // 5]
valid_subjects = subjects[len(subjects) * 4 // 5:]

train_anns = [ann for ann in anns if ann['subject'] in train_subjects]
valid_anns = [ann for ann in anns if ann['subject'] in valid_subjects]

with (dst_dir / 'train_anns.json').open('w') as f:
    json.dump(train_anns, f, indent=2, ensure_ascii=False)
with (dst_dir / 'valid_anns.json').open('w') as f:
    json.dump(valid_anns, f, indent=2, ensure_ascii=False)