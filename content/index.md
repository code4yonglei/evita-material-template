# High Performance Data Analytics in Python

Scientists, engineers and professionals from many sectors are seeing an enormous growth in the size and number of datasets relevant to their domains. Professional titles have emerged to describe specialists working with data, such as data scientists and data engineers, but also other experts are finding it necessary to learn tools and techniques to work with big data. Typical tasks include preprocessing, analysing, modeling and visualising data.

Python is an industry-standard programming language for working with data on all levels of the data analytics pipeline. This is in large part because of the rich ecosystem of libraries ranging from generic numerical libraries to special-purpose and/or domain-specific packages, often supported by large developer communities and stable funding sources.

This lesson will give an overview of working with research data in Python using general libraries for storing, processing, analysing and sharing data. The focus is on high performance. After covering tools for performant processing on single workstations the focus shifts to profiling and optimising, parallel and distributed computing.





:::{prereq}

- Basic experience with Python
- Basic experience in working in a Linux-like terminal
- Some prior experience in working with large or small datasets

:::



## Reading materials

- [Python for Scientific Computing](https://aaltoscicomp.github.io/python-for-scicomp/)
- [Using Python in an HPC Environment](https://uppmax.github.io/HPC-python/)
- [Python Performance Workshop](https://enccs.github.io/python-perf/)
- ...



```{toctree}
:caption: Software setup
:maxdepth: 1

Python-HPDA-0-SoftwareSetup
```





```{toctree}
:caption: Lesson episodes
:maxdepth: 1

Python-HPDA-1-Motivation
Python-HPDA-2-EfficientArrayComputing
3-jupyter-notebook-styling
```



## Learning outcomes

This material is for all researchers and engineers who work with large or small datasets and who want to learn powerful tools and best practices for writing more performant, parallelised, robust and reproducible data analysis pipelines.

By the end of a workshop covering this lesson, learners should:
- Have a good overview of available tools and libraries for improving performance in Python (**link to leaves in skill tre**)
- Knowing libraries for efficiently storing, reading and writing large data  (**link to leaves in skill tree**)
- Be comfortable working with NumPy arrays and Pandas dataframes for data analysis using Python (**link to leaves in skill tree**)
- ...


## Instructor’s guide

### Teaching hours and number of participants

This module is developed for one instructor and two teaching assistants during teaching for 40 students. The teaching can be delivered online, onsite, or in a hybrid format, providing flexibility to accommodate different learning preferences and circumstances. Whether students attend in person or remotely, the course materials and exercises are designed to ensure a consistent and engaging learning experience. Instructors can leverage virtual tools such as video conferencing and shared coding environments alongside traditional classroom setups to support all modes of delivery effectively.

Students are expected to dedicate 2-4 hours in total. This estimate combines contact hours, such as lectures or exercises (1~2 hours), with independent learning time (1-2 hours), which include self-study, assignments, and revision. Balancing these components ensures students have sufficient guided instruction while allowing ample time for personal engagement with the material.


### Mode of teaching and exercising

Before teaching this module, the instructor is expected to set up the programming environment on the HPC cluster or assist students with installing the necessary packages on their personal computers. During exercises, instructors may use Jupyter notebooks for demonstrations or copy and paste code examples from the webpage into script files. These scripts can then be executed on students’ personal computers or the HPC cluster, providing a flexible learning experience.

### Hardware requirements on HPC clusters

The HPC cluster used for this course should be equipped with modern NVIDIA GPUs that support CUDA programming. It is recommended that the cluster allocates a minimum of 20 GPU hours per person throughout the course duration. This allocation ensures sufficient time for code development, testing, and running computationally intensive workloads. A minimum of 16 GB GPU memory is recommended to handle large datasets and complex computations efficiently. 




```{admonition} Credit
:class: warning

Don't forget to check out additional [**course materials**](https://www.evitahpc.eu/) from XXX. Please [**contact us**](https://www.evitahpc.eu/) if you want to reuse these course materials in your teaching. You can also join the [**XXX channel**](https://www.evitahpc.eu/) to share your experience and get more help from the community.


```



```{admonition} Licensing
:class: danger

Copyright © 2025 XXX. This material is released by XXX under the Creative Commons Attribution 4.0 International (CC BY 4.0). These materials may include references to hardware and software developed by other entities; all applicable licensing and copyrights apply.
```

