# ============================================================
# Script Name:   md_episodes.py
# Description:   This file is used to update Markdown files for
#                `Software setup` and `Lesson` episodes.
#                - remove `Content of this notebook`
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



def convert_directives(input_file):

	output = []
	code_block = []
	is_code_block = False

	for line in input_file:
		if line.startswith('<font color='):
			is_code_block = True
			if code_block:
				code_block = []
			code_block.append(line)
		elif is_code_block == True:
			code_block.append(line)
		else:
			output.append(line)

		# process code block
		if line.startswith("</div>") and is_code_block == True:
			is_code_block = False

			# check if there is a title
			title_lists = code_block[0].split("'>**")
			title_lists = title_lists[1]
			title_lists = title_lists.split("**</font>")
			if len(title_lists) > 1:
				keyword = title_lists[0]
				title = title_lists[1]
				if ":" in title and 'exercise' in keyword.lower():
					output.append(":::{" + keyword.lower() + "}" + title[1:])
					output.append(":class: dropdown\n\n")
				elif "quiz" in keyword.lower():
					output.append(":::{exercise} Homework\n")
					output.append(":class: dropdown\n\n")
				elif "References" in keyword:
					output.append(":::{seealso}\n")
				else:
					output.append(":::{" + keyword.lower() + "}" + title)
			else:
				keyword = title_lists[0]
				output.append(":::{" + keyword.lower() + "}")
			for line in code_block[1:-1]:
				output.append(line)
			output.extend(":::\n\n\n")

	return output



def process_solution_directive(input_file):

	output = []
	code_block = []
	is_code_block = False

	for line in input_file:
		if line.startswith('<details>'):
			is_code_block = True
			if code_block:
				code_block = []
			code_block.append(line)
		elif is_code_block == True:
			code_block.append(line)
		else:
			output.append(line)

		# process code block
		if line.startswith("</details>") and is_code_block == True:
			is_code_block = False

			# check if there is a title

			output.append(":::{solution}\n")
			output.append(":class: dropdown\n")
			for line in code_block[3:-1]:
				output.append(line)
			output.extend(":::\n\n\n")

	return output



def update_video_format(input_file):
	"""
	update styling format for video
	"""
	output = []
	video_block = []
	is_video_block = False

	for line in input_file:
		if line.startswith('<iframe'):
			is_video_block = True
			if video_block:
				video_block = []
			video_block.append(line)
		elif is_video_block == True:
			video_block.append(line)
		else:
			output.append(line)

		# process code block
		if line.startswith("></iframe>") and is_video_block == True:
			is_video_block = False
			video_frame = ""
			for line in video_block:
				video_frame += line.strip('\n').lstrip() + ' '
			output.append(video_frame)
			output.append('\n\n\n')

	return output





def update_image_format(input_file):
	"""
	update styling format for images
	"""
	output = []
	for line in input_file:
		if "<img" in line:
			image_info = line.split('"')
			image_path = image_info[1]
			size = image_info[3]
			sphinx_image = "![](" + image_path + ")"
			output.append(sphinx_image + '\n\n')
		else:
			output.append(line)

		if "library" in line:
			print(line)
			output.append(line)
	return output



def remove_toc(input_file):
	output = []
	skip = False
	for line in input_file:
		if line.startswith("#### Content of this notebook"):
			skip = True  # Start skipping lines
			continue  # skip the starting line
		if skip and line.startswith("## "):
			skip = False  # Stop skipping after this line
			output.append(line)
			continue  # skip the ending line
		if not skip:
			output.append(line)

	return output





def process_episodes_md(input_file, output_file):
	"""
	update styling settings from raw markdown files
	"""
	# remove stylings in Jupyter NB
	contents = []
	styling_begin = '<div class="alert alert-'
	styling_end = 'test' # '</div>'

	with open(input_file, 'r', encoding='utf-8') as f:
		for line in f:
			if styling_begin in line or styling_end in line:
				continue
			# remove `#### Learning objectives` and `#### Instructor notes`
			if '#### Learning objectives' in line or '#### Instructor notes' in line:
				continue
			contents.append(line)

	# remove toc
	contents = remove_toc(contents)
	contents = convert_directives(contents)
	contents = process_solution_directive(contents)
	contents = update_video_format(contents)
	contents = update_image_format(contents)

	with open(output_file, 'w', encoding='utf-8') as f_out:
		f_out.writelines(contents)	



if __name__ == "__main__":
	input_file = "Python-HPDA-2-EfficientArrayComputing-raw.md"
	process_episodes_md(input_file, '000000000000.md')
	print(f"Deleted alert-info blocks from {input_file}")

