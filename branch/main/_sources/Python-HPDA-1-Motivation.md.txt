# 1. Motivation

This episode provides a broad overview of this course and the main motivation to attend this course.

```{objectives}
- What is big data
- What is the Python programming environment and the ecosystem
- What you will learn during this course
```

```{instructor-note}
- 20 min teaching/type-along
-  0 min exercising
```

## 1.1 Big Data

:::{discussion}: How large is your data?

- How large is the data you are working with?
- Are you experiencing performance bottlenecks when you try to analyse it?
:::



“Big data refers to data sets that are too large or complex to be dealt with by traditional data-processing application software. […] Big data analysis challenges include capturing data, data storage, data analysis, search, sharing, transfer, visualization, querying, updating, information privacy, and data source.” (from [Wikipedia](https://en.wikipedia.org/wiki/Big_data))

“Big data” is a current buzzword used heavily in the tech industry, but many scientific research communities are increasingly adopting high-throughput data production methods which lead to very large datasets. One driving force behind this development is the advent of powerful machine learning methods which enable researchers to derive novel scientific insights from large datasets. Another is the strong development of high performance computing (HPC) hardware and the accompanying development of software libraries and packages which can efficiently take advantage of the hardware.

This course focuses on high-performace data analytics (HPDA), a subset of high-performance computing which focuses on working with large data. The data can come from either computer models and simulations or from experiments and observations, and the goal is to preprocess, analyse and visualise it to generate scientific results.

The video shown below provide more descriptions of the big data.


```python
from IPython.display import YouTubeVideo

YouTubeVideo('qydP7cOH4qc', width=600, height=360)
```





<iframe width="600" height="360" src="https://www.youtube.com/embed/qydP7cOH4qc" frameborder="0" allowfullscreen  ></iframe> 






## 1.2 Python

:::{discussion}

**Performance bottlenecks in Python**
- Have you ever written Python scripts that look something like the one below?
- Compared to C/C++/Fortran, this for-loop will probably be orders of magnitude slower

```python
f = open("mydata.dat", "r")
for line in f.readlines():
    fields = line.split(",")
    x, y, z = fields[1], fields[2], fields[3]
    # some analysis with x, y and z
f.close()
```

:::



Despite early design choices of the Python language which made it significantly slower than conventional HPC languages, a rich and growing ecosystem of open source libraries have established Python as an industry-standard programming language for working with data on all levels of the data analytics pipeline. These range from generic numerical libraries to special-purpose and/or domain-specific packages. This lesson is focused on introducing modern packages from the Python ecosystem to work with large data. Specifically, we will learn to use:
- Numpy
- Scipy
- Pandas
- Xarray
- Numba
- Cython
- multithreading
- multiprocessing
- Dask

## 1.3 What You Will Learn

This lesson provides a broad overview of methods to work with large datasets using tools and libraries from the Python ecosystem. Since this field is fairly extensive we will not have time to go into much depth. Instead, the objective is to expose just enough details on each topic for you to get a good idea of the big picture and an understanding of what combination of tools and libraries will work well for your particular use case.

Specifically, **this course covers**:
- Tools for efficiently storing data and writing/reading data to/from disk
- How to share datasets and mint digital object identifiers (DOI)
- Main methods of efficiently working with tabular data and multidimensional arrays
- How to measure performance and boost performance of time consuming Python functions
- Various methods to parallelise Python code

The course does not cover the following episodes but the lesson materials for these episodes are provided at other modules. Please refer to the links provided below to the other course materials. 
- [Visualisation techniques](https://enccs.github.io/gpu-programming/)
- [Machine learning](https://enccs.github.io/gpu-programming/)
- [GPU programming](https://enccs.github.io/gpu-programming/)

## 1.4 Keypoints

- Datasets are getting larger across nearly all scientific and engineering domains
- The Python ecosystem has many libraries and packages for working with big data efficiently
