# ============================================================
# Script Name:   md_index.py
# Description:   This file is used to convert main Markdown
#                 file to `index.md` file.
#
# Author:        Yonglei WANG
# Date:          2025-08-12
# Version:       1.0
# Usage:         
# Notes:         Ensure input file is UTF-8 encoded.
# ============================================================
#
#!/usr/bin/env python3
#-*- coding: utf-8 -*-


softwareSetup_episode = []
lesson_episodes = []


def add_stryling_for_sphinx(color, title, input_content):
	"""
	add styling setting for sphinx rendering
	"""
	output = []
	color_directive = {'red': 'danger', 'orange': 'warning'}
	if color in color_directive:
		output.append("\n\n```{admonition} " + title + '\n')
		output.append(":class: " + color_directive[color] + '\n')

	for line in input_content[1:]:
		output.append(line)
	output.append("```\n\n")
	return output



def convert_prerequisites(input_content):
	"""
	convert title of "## Prerequisites" to `prereq` structure
	"""
	output = []
	output.append("\n\n\n\n:::{prereq}\n")

	for line in input_content[1:]:
		output.append(line)
	output.append(":::\n\n\n\n")
	return output



def convert_toctree_structure(item, input_content):
	"""
	convert titles of "## Software setup" and "## Lesson episodes"
	to `toctree` structure
	"""
	output = []
	output.append("\n\n```{toctree}\n")
	output.append(":caption: " + item + "\n")
	output.append(":maxdepth: 1\n\n")

	ipynb_titles = []
	for line in input_content:
	    if '](./' in line:
	        line = line.strip()
	        start_idx = line.find("](./")
	        end_idx = line.find(".ipynb")
	        if start_idx != -1 and end_idx != -1:
	            ipynb_title = line[start_idx+4:end_idx]
	            ipynb_titles.append(ipynb_title + "\n")
	for line in ipynb_titles:
		output.append(line)
		if "Lesson episodes" in item:
			lesson_episodes.append(line)
		if "Software setup" in item:
			softwareSetup_episode.append(line)
		print(line)
	if "Lesson episodes" in item:
		output.append('3-jupyter-notebook-styling\n')
	output.append("```\n\n\n\n")
	return output



def get_title_intro_part(input_file, output_file):
	"""
	Copy lines starting from a line beginning with `# ` 
	until just before the next line starting with '## '
	"""
	sections = []
	section = []

	for line in input_file:
		if line.startswith('# ') or line.startswith("## "):
			if section:
				sections.append(section)
			section = []
			section.append(line)
		else:
			section.append(line)
	sections.append(section)

	for index, section in enumerate(sections):
		# convert `Prerequisites` section
		if "## Prerequisites" in section[0]:
			section = convert_prerequisites(section)
			sections[index] = section
		# convert `Software setup` and `Lesson episodes` sections
		if "## Software setup" in section[0]:
			section = convert_toctree_structure('Software setup', section)
			sections[index] = section
		if "## Lesson episodes" in section[0]:
			section = convert_toctree_structure('Lesson episodes', section)
			sections[index] = section
		# add settings for sphinx rendering
		if "## Credit" in section[0]:
			section = add_stryling_for_sphinx('orange', 'Credit', section)
			sections[index] = section
		if "## Licensing" in section[0]:
			section = add_stryling_for_sphinx('red', 'Licensing', section)
			sections[index] = section

	with open(output_file, 'w', encoding='utf-8') as f:
		for section in sections:
			f.writelines(section)



def create_index_md(input_file, output_file):
	"""
	remove styling settings from raw markdown files
	"""
	styling_begin = '<div class="alert alert-'
	styling_end = '</div>'

	content_remove_styling = []
	with open(input_file, 'r', encoding='utf-8') as f:
		for line in f:
			if styling_begin in line or styling_end in line:
				continue
			content_remove_styling.append(line)

	get_title_intro_part(content_remove_styling, 'index.md')

