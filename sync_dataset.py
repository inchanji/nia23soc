from glob import glob
import os
import shutil
import sys


base_dir = ['원천데이터', '라벨링데이터']

tgt_dir = '/home/disk2/nia23soc'
src_dir = './nas_pcn'

def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

filelist = []
dir_list = []
# include subdirectories
for dir in base_dir:
    dir_list += fast_scandir(os.path.join(src_dir, dir))
for dir in dir_list:
    filelist += [ f for f in glob(os.path.join(dir, '*')) if os.path.isfile(f)]
    # print(os.path.join(dir, '*'))
    # print(glob(os.path.join(dir, '*')))


# copy and paste to target directory
for path in filelist:
    #check if file exists
    path_tgt = path.replace(src_dir, tgt_dir)
    if not os.path.exists(path_tgt):
        os.makedirs(os.path.dirname(path_tgt), exist_ok=True)
        shutil.copy(path, path_tgt)
        print('File copied: ', path_tgt)
        pass
    else:
        print('File already exists: ', path_tgt)
        # sys.exit(1)

    # shutil.copy(path, os.path.join(tgt_dir, os.path.basename(path)))

print(len(filelist))