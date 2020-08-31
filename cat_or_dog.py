
import pandas as pd
from fastai.vision import untar_data, ImageDataBunch, cnn_learner, accuracy, URLs, get_transforms, imagenet_stats,models
from fastai.vision import load_learner, ImageList, DatasetType
import fastai
from os import listdir

from fastprogress.fastprogress import force_console_behavior
import fastprogress
fastprogress.fastprogress.NO_BAR = True
master_bar, progress_bar = force_console_behavior()
fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar



def get_file(aString):
    return str(aString.split('/')[-1])

image_folder = 'images/'

path = untar_data(URLs.DOGS)
learn = load_learner(path, test=ImageList.from_folder(image_folder), bs = 1)
preds,y = learn.get_preds(ds_type=DatasetType.Test, )
predList = list(preds.numpy()[:,0])

f_names = listdir(image_folder)

pred_df = pd.DataFrame(list(zip(f_names,predList)), columns = ['f_name','prob_dog'])

registry = 'registry/downloaded_files.csv'
regDF = pd.read_csv(registry)


regDF['f_name'] = regDF.file.apply(get_file)

out_df = pd.merge(regDF,pred_df, on = ['f_name'])

out_df.to_csv(f'registry/cat_dog_pred.csv', index = False)