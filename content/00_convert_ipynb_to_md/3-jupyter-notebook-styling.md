# 3. Sphinx Directives

## Directives

There are directives in Sphinx to highlight code blocks
- `discussion`, `demo`, `exercises`, `solution`, `homework`, `seealso`...
- `note`, `hint`, `important`, `attention`, `caution`,`warning`, `danger`, ...

In Jupyter NB, there is no such stryling settings. We can ask course developers to provide some keywords (as listed below) before the code block. and we can convert these Jupyter NB blocks to Sphinx blocks.

The format in the Jupyter NB should look like the content below:
```
<font color='purple'>**Discussion**</font>: How large is your data?

- How large is the data you are working with?
- Are you experiencing performance bottlenecks when you try to analyse it?
</div>
```

- Highlight the keyword (here it is `Discussion`) using the corresponding font color
- the Jupyter NB code block starts with `<font color=` and ends with `</div>`


```{discussion} How large is your data?

- How large is the data you are working with?
- Are you experiencing performance bottlenecks when you try to analyse it?
```


```{attention}

Do not remove the following line.
```



```{caution}

Do not remove the following line.
```


```{warning}

Do not remove the following line.
```


```{danger}

Here is a danger!
```


```{error}

Here is an error!
```


```{hint}

In this exercise, you can use A instead of B.
```


```{tip}

In this exercise, you can use A instead of B.
```

```{important}

You should use the latest version of Python.
```


```{exercise}

Description of exercise

- Do this
- then do this
- finally observe what happens when you do this…do this…
```


```{exercise} With an option "Click to show"
:class: dropdown

Description of exercise
- Do this
- then do this
- finally observe what happens when you do this…do this…
```


```{solution}

Here is the solution for above exercises.
```


```{homework}

Here are the homework assignments for this episode.
```


```{note}

Here we chose to summarize the data by its mean, but many other common statistical functions are available as dataframe methods, like `std()`, `min()`, `max()`, `cumsum()`, `median()`, `skew()`, `var()`, *etc.*
```


```{demo}

Code for demonstration
```


```{seealso}

publications and webpages for references
```


## Define addition keywords

- `suggestion`, `recommendation`, ...


```{callout} Suggestion

It is recommended to use the CUDA version > 10.0.
```


## Badges

Here is one badge after we publish this module at Zenodo.

:::{image} https://zenodo.org/badge/DOI/10.5281/zenodo.14844443.svg
  :target: https://doi.org/10.5281/zenodo.14844443
:::


We can design other badges with different colors.

```{image} https://img.shields.io/badge/AAAA-blue?style=plastic
```

```{image} https://img.shields.io/badge/BBBB-purple?style=plastic
```

```{image} https://img.shields.io/badge/CCCC-red?style=plastic
```

```{image} https://img.shields.io/badge/DDDD-996300?style=plastic
```

```{image} https://img.shields.io/badge/EE EE-gold?style=plastic
```

```{image} https://img.shields.io/badge/FF FF-orange?style=plastic
```

```{image} https://img.shields.io/badge/Introductory-green?style=plastic
```

```{image} https://img.shields.io/badge/Intermediate-blueyellow?style=plastic
```

<iframe width="560" height="315" src="https://www.youtube.com/embed/YB-LCJBRvFs?start=4180" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

