# ============================================================
# Script Name:   main.py
# Description:   These Python scripts convert raw Markdown files
#                (exported from Jupyter notebooks) into structured
#                Markdown files, ready for rendering into a
#                Sphinx-based webpage.
#
# Author:        Yonglei WANG
# Date:          2025-08-12
# Version:       1.0
# Usage:         python3 main.py Python-HPDA-raw.md
# Notes:         Ensure input file is UTF-8 encoded.
# ============================================================
#
#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import sys

import md_index
import md_episodes

# create the index file from the main NB, here it is `Python-HPDA-raw.md`
main_input_file = sys.argv[1]
index_file = 'index.md'
md_index.create_index_md(main_input_file, index_file)

# process `software setup` episode
print(md_index.softwareSetup_episode)
softwareSetup_raw = md_index.softwareSetup_episode[0][:-1] + '-raw.md'
softwareSetup_out = md_index.softwareSetup_episode[0][:-1] + '.md'
md_episodes.process_episodes_md(softwareSetup_raw, softwareSetup_out)

# process `lesson` episodes
# print(md_index.lesson_episodes)
for ifile in md_index.lesson_episodes:
	print(ifile)

ifile = 'Python-HPDA-1-Motivation'
episode_raw = ifile + '-raw.md'
episode_out = ifile + '.md'
print(episode_raw, episode_out)
md_episodes.process_episodes_md(episode_raw, episode_out)


ifile = 'Python-HPDA-2-EfficientArrayComputing'
episode_raw = ifile + '-raw.md'
episode_out = ifile + '.md'
print(episode_raw, episode_out)
md_episodes.process_episodes_md(episode_raw, episode_out)

