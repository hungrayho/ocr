import argparse
import cv2
import numpy as np
from numpy import empty
import pandas as pd
from matplotlib import pyplot as plt
import os

def map_split(bbox_map_1, bbox_map_2):
    bbox_map_split_1 = []
    bbox_map_split_2 = []
    for i in bbox_map_1:
        bbox_map_split_1.append(i.split())
    for i in bbox_map_2:
        bbox_map_split_2.append(i.split())     
    return bbox_map_split_1, bbox_map_split_2 

def gen_df(bbox_map_split_1, bbox_map_split_2):
    df_1 = pd.DataFrame(bbox_map_split_1, columns=['ENTRY_FIELD', 'TYPE', 'CONTEXT', 'FIELD_DESC', 'XMIN', 'YMIN', 'XMAX', 'YMAX'])
    df_1['PAGE'] = 1
    df_2 = pd.DataFrame(bbox_map_split_2, columns=['ENTRY_FIELD', 'TYPE', 'CONTEXT', 'FIELD_DESC', 'XMIN', 'YMIN', 'XMAX', 'YMAX'])
    df_2['PAGE'] = 2
    df = df_1.append(df_2)
    df = df.reset_index()
    df['ENTRY'] = empty((len(df), 0)).tolist()
    return df

def field_extract(df):
    for i in range (0, len(df)):
        if df["PAGE"][i] == 1:
            df["ENTRY"][i] = img1[int(df["YMIN"][i]):int(df["YMAX"][i]), int(df["XMIN"][i]):int(df["XMAX"][i])]
        if df["PAGE"][i] == 2:
            df["ENTRY"][i] = img2[int(df["YMIN"][i]):int(df["YMAX"][i]), int(df["XMIN"][i]):int(df["XMAX"][i])]
    print(df.head())
    plt.imshow(df["ENTRY"][3], cmap='gray')
    plt.show()
    return df

#scaling & rotating the tax documents

def rescale(directory,file,template_path):
    num_features = 7500
    match_threshold = 0.1
    
    input_path = str(os.path.join(directory,file))
    
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    
    orb = cv2.ORB_create(num_features)
    keypoints1, descriptors1 = orb.detectAndCompute(img, None)
    keypoints2, descriptors2 = orb.detectAndCompute(template, None)
    
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    matches.sort(key=lambda x: x.distance, reverse=False)
    matches = matches[:int(len(matches) * match_threshold)]
    
    #visual representation of matched features for scaling and rotation
    #imMatches = cv2.drawMatches(img, keypoints1, template, keypoints2, matches, None)
    #cv2.imwrite("matches.jpg", imMatches)
    
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width = template.shape
    img_warp = cv2.warpPerspective(img, h, (width, height))
    
    #generate temporary image
    cv2.imwrite(file, img_warp)
    
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #positional args
    parser.add_argument(
            '--directory', nargs='?',
            type=str,
            required=False,
            default=os.getcwd(),
            help='Path to Tax reform documents'
            )
    parser.add_argument(
            '--File1', nargs='?',
            type=str,
            required=False,
            default='test.png',
            help='Tax reform document page 1'
            )
    parser.add_argument(
            '--File2', nargs='?',
            type=str,
            required=False,
            default='test2.png',
            help='Tax reform document page 2'
            )
    parser.add_argument(
            '--fmap1', nargs='?',
            type=str,
            required=False,
            default='fieldmap_1040_1988_1',
            help='path to field map 1'
            )
    parser.add_argument(
            '--fmap2', nargs='?',
            type=str,
            required=False,
            default='fieldmap_1040_1988_2',
            help='path to field map 2'
            )
    parser.add_argument(
            '--output', nargs='?',
            type=str,
            required=False,
            default='fields.csv',
            help='output path for extracted fields csv'
            )

    FLAGS = parser.parse_args()
    
   
    rescale(FLAGS.directory,FLAGS.File1,"data/1040_1988_1.png")
    rescale(FLAGS.directory,FLAGS.File2,"data/1040_1988_2.png")
    
    img1 = cv2.imread(FLAGS.File1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(FLAGS.File2, cv2.IMREAD_GRAYSCALE)
    
    with open('data/fieldmap_1040_1988_1') as f:
        bbox_map_1 = f.read().splitlines()
    with open('data/fieldmap_1040_1988_2') as f:
        bbox_map_2 = f.read().splitlines()
    
    bbox_map_split_1, bbox_map_split_2 = map_split(bbox_map_1, bbox_map_2)
    
    df = gen_df(bbox_map_split_1, bbox_map_split_2)
    df = field_extract(df)
    df.to_csv(FLAGS.output)
    print (f"Fields successfully extracted to \"{FLAGS.output}\"")
