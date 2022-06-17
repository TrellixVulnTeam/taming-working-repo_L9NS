
import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from lsun import LSUNCatsTrain,LSUNCatsValidation
from tqdm import tqdm

lsun_train=LSUNCatsTrain()
lsun_val=LSUNCatsValidation()


for lsun in [lsun_train,lsun_val]:
    filenames = lsun.image_paths
    clean_files=[]
    for i in tqdm(range(len(lsun))):
        img=lsun[i]
        if img is None:
            print('File {} is corrupted, excluding it.'.format(filenames[i]))
        else:
            clean_files.append(filenames[i])
    with open('{}_cleaned.txt'.format(lsun.data_paths[:-4]),'w') as fw:
        fw.write('\n'.join(clean_files))


