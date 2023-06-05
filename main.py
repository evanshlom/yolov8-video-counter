import pandas as pd

from model import predict
from class_ids import class_dict
from glob2 import glob
from statistics import median

predict("video.mp4")
print("bounding boxes predicted")

labels = pd.concat([pd.read_csv(path, sep=' ', header=None).assign(frame=i) for i, path in enumerate(glob('runs\detect\predict\labels\*.txt'))])
print(labels.shape)

video_duration = 248 # seconds (video duration is 4 minutes 8 seconds)

labels['class'] = labels[0].map(class_dict)
labels = labels[labels['class'] == 'person']
labels = labels.drop(columns='class')

labels = labels.drop(columns=[1, 2, 3, 4, 5])

labels[1] = 1
labels = labels.rename(columns={1: 'count'})

labels = labels.groupby('frame')['count'].sum()
labels = pd.DataFrame(labels)

second = 1
labels['second'] = 1
for i in range(0, len(labels)):
    labels['second'][i] = second
    if (i+1) % 12 == 0: # every 12th row because ~12 fps (12.5, rounding down because goal is generalizing statistic across all frames) video duration is 4:08
        second += 1

# get dict of highest value in count column for each value in second column, in format of {second: max count}
median_persons_detected = {second: int(median(labels['count'][labels['second'] == second])) for second in labels['second']}
# apply dict to labels df
labels['persons detected'] = labels['second'].map(median_persons_detected)

print(labels.head(35))
print(f'''
      The most people detected during the video:, {max(labels['persons detected'])}
      The lowest number of people detected:, {min(labels['persons detected'])}
      *The average number of people detected:, {sum(labels['persons detected'])/len(labels['persons detected'])}
      ''')