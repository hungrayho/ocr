# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:04:26 2019

@author: Ray
"""
import argparse
import cv2
from numpy import empty
import pandas as pd
from matplotlib import pyplot as plt
#import time

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #positional args
    parser.add_argument(
            '--img1', nargs='?',
            type=str,
            required=False,
            default='test.png',
            help='path to img 1'
            )
    parser.add_argument(
            '--img2', nargs='?',
            type=str,
            required=False,
            default='test2.png',
            help='path to img 2'
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

    img1 = cv2.imread(FLAGS.img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(FLAGS.img2, cv2.IMREAD_GRAYSCALE)
    
    with open(FLAGS.fmap1) as f:
        bbox_map_1 = f.read().splitlines()
    with open(FLAGS.fmap2) as f:
        bbox_map_2 = f.read().splitlines()
    
    bbox_map_split_1, bbox_map_split_2 = map_split(bbox_map_1, bbox_map_2)
    
    df = gen_df(bbox_map_split_1, bbox_map_split_2)
    df = field_extract(df)
    df.to_csv(FLAGS.output)
    print (f"Fields successfully extracted to \"{FLAGS.output}\"")