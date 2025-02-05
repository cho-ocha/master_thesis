# -*- coding: utf-8 -*-
#第一引数のディレクトリ内のからを取り出す

import re
import json
import sys
import glob
import os
import argparse


# parserを作成
parser = argparse.ArgumentParser()
parser.add_argument('dir_path', help='designate dir path')
args = parser.parse_args()
  
# ファイル名のみを取得
pathlist = glob.glob(args.dir_path + '/*.txt')

for path in pathlist:
#    print(path)
#    print(path.lstrip(args.dir_path))
    print('python3.8 mecab_kara.py ' + path)
