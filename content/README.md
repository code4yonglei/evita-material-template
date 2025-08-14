# From Jupyter Notebooks to Markdown files and then Sphinx rendering


## Creating a GitHub repository

- Go to [Sphinx template](https://github.com/ENCCS/sphinx-lesson-template)
- Click the button `Use this template` to `Create a new repository`
- Create static webpage from this GitHub repository
- Go to `evita-material-template/content` directory
	- remove `guide.md`, `guide.rst-if-wanted`, `index.md`, `index.rst-if-wanted`, `quick-reference.md` and `quick-reference.rst-if-wanted`
	- in practice you have a `conf.py` file and two directories (`_static_` and `img`) in the `content` folder


## Materials from course developer(s)

- Course developers should provide relevant materials
	- a `module-ipynb` directory containing all Jupyter notebooks
		- `Python-HPDA.ipynb`, this is the main Jupyter NB for this module
		- `Python-HPDA-0-SoftwareSetup.ipynb`, this is the Jupyter NB to setup programming environment
		- `Python-HPDA-1-Motivation.ipynb`, this is one Jupyter NB for lecturing
		- `Python-HPDA-2-EfficientArrayComputing.ipynb`, this is another Jupyter NB for lecturing
	- a `module-code` directory containing all code examples for this module
	- a `module-images` directory containing all images for this module
- Update `conf.py` file with practical info for this module
	- replace `LESSON NAME` with `High Performance Data Analytics in Python`
	- replace `copyright = "2021, The contributors"` with something
	- replace `author = "The contributors"` with something
	- replace `github_user = "ENCCS"` with something


## Processing Jupyter NBs to Markdown files

- put the logo file `evita.png` at `img` directory
- create a `00_convert_ipynb_to_md` directory with correpondent files
	- four Python scripts in this directory
	- `cd 00_convert_ipynb_to_md/`
	- run `jupyter nbconvert ../module-ipynb/Python*.ipynb --to markdown` to all Jupyter NBs to raw Markdown documents
	- run `cp ../module-ipynb/Python*.md .`
	- run `python3 rename_md.py Python` to rename all Markdown files
	- run `python3 main.py Python-HPDA-raw.md` with the main Jupyter NB as an argument
	- run `cp Python*[!-raw].md index.md 3-*.md ../`
- upload relevant files to the repository and commit changes

