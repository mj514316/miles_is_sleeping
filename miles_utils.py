import tweepy
import pandas as pd
import wget

import os
import sys
sys.path.append("fast-style-transfer/")
sys.path.append("fast-style-transfer/src")
from evaluate import ffwd_different_dimensions, ffwd
from utils import list_files
import random

def authorize_twitter():
    auth = tweepy.OAuthHandler(os.getenv('consumer_token'), os.getenv('twitter_consumer_secret'))
    auth.set_access_token(os.getenv('twitter_key'),os.getenv('twitter_secret_key'))
    return tweepy.API(auth)

def getMilesTweets(dest_folder = 'images', registry = 'registry/downloaded_files.csv'):

    api = authorize_twitter()
    timeline = api.user_timeline(screen_name='@MilesSleeping')
    
    registryDF = pd.read_csv(registry)
    downloadedFiles = registryDF['file'].to_list()
    
    media_files = set()
    for status in timeline:
        media = status.entities.get('media',[])
        if(len(media) > 0):
            media_files.add((media[0]['media_url'],status.id))
            
    for media_file in media_files:
        fname = media_file[0]
        if fname not in downloadedFiles:
            print(f'downloading {fname}')
            wget.download(media_file[0],dest_folder)
        else:
            pass
        
    newFiles = pd.DataFrame(media_files, columns = ['file','tweet_id'])
    pd.concat([newFiles,registryDF]).drop_duplicates().to_csv(registry, index = False)
    return media_files

def random_style_transfer(in_path = 'images/', out_path = 'processed_images/', checkpoint_path = 'checkpoints', allow_different_dimensions = True, batch_size = 1, device = '/gpu:0'):
    checkpoints = list_files(checkpoint_path)
    files = [fname for fname in list_files(in_path) if fname not in list_files('processed_images/')]
    fullprocess = [(os.path.join(in_path,x),os.path.join(out_path,x),f'{checkpoint_path}/{random.sample(checkpoints,1)[0]}') for x in files]
    for tup in fullprocess:
        print(tup)
        if allow_different_dimensions:
            ffwd_different_dimensions([tup[0]], [tup[1]], tup[2], 
                    device_t=device, batch_size=batch_size)
        else :
            ffwd([tup[0]], [tup[1]], tup[2], device_t=device,
                    batch_size=batch_size)