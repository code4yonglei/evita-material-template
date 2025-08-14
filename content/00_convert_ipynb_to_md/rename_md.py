# ============================================================
# Script Name:   rename_md.py
# Description:   This Python script rename Markdown files
#                (exported from Jupyter notebooks) into raw
#                Markdown files.
#
# Author:        Yonglei WANG
# Date:          2025-08-12
# Version:       1.0
# Usage:         python3 rename_ipynb.py Python
# Notes:         Ensure input file is UTF-8 encoded.
# ============================================================
#
#!/usr/bin/env python3
#-*- coding: utf-8 -*-


import os
import sys

file_beginning_string = sys.argv[1]

folder = "."

for filename in os.listdir(folder):
    if filename.startswith(file_beginning_string):
        old_path = os.path.join(folder, filename)

        new_name = filename[:-3] + "-raw.md"
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)
        print(f'Renamed: {filename} ---> {new_name}')
