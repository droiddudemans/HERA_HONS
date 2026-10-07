# hera_notebook_templates
Repository for HERA analysis / real-time pipeline jupyter notebooks and related
support code

# Installation
Install using `pip install .` from within the head directory. If you plan to do
development on this code base, install using `pip install -e .`  instead.

# Usage

## Notebook Templates

The `notebooks/` directory has several notebook templates that can be run manually 
(or interactively). These generally take in "parameters" as environment variables.

## CLI (preferred)

The `hera_notebook_templates` package also provides a CLI for running notebooks that
are saved into the `hera_notebook_templates/notebooks/` directory. We expect that most
notebooks will be migrated to this location over time. When using the CLI, the notebooks
are executed using `papermill` and the parameters are passed in as command line arguments.
This has several advantages, for example, the notebooks are found automatically and you
don't need to specify the path to the repo. `papermill` also tracks progress of the 
execution, and handily makes save-points of the notebook along the way.

To use the CLI, invoke the `hnote` command:

```
hnote --help
```

This will print out help information. To print out all available notebooks, use

```
hnote avail
```

To run a notebook, use the `run` command:

```
hnote run <notebook_name> <param1>=<value1> <param2>=<value2> ...
```

To get a list of the available parameters for a notebook, use the `inspect` command 
(recall that this is only applicable to the notebooks inside the package, not all
notebooks in the repo, but this will change over time):

```
hnote inspect <notebook_name>
```
