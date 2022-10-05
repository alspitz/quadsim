# quadsim

## Setup

First, install dependencies.

Make sure you have numpy, scipy, and matplotlib (for Python 3).
(Can install these using your system's pacakage manager).

Install meshcat e.g. with `pip install meshcat` (required for visualization).

Quadsim requires the `python_utils` repo to exist on a `PYTHONPATH`.

The recommended setup is to create a folder named "python" (e.g. in your home folder) and then clone both python_utils and quadsim in `~/python`.
Note that cloning `quadsim_devel` requires access permissions.

```
mkdir ~/python
cd ~/python
git clone https://github.com/alspitz/python_utils
git clone https://github.com/alspitz/quadsim_devel quadsim
```

Next, in your `.bashrc`, add `${HOME}/python` to `PYTHONPATH`
e.g. add the following line.
`export PYTHONPATH="${HOME}/python":"${PYTHONPATH}"`

Change directory and rc file as needed (e.g. if using zsh).

## Usage

Run `python main.py` and `python compare.py`.

A browser window should open with a visualization of the quadrotor executing the reference trajectories with the specified controllers.

After that, plots should appear showing state and control variables.

`compare.py` has some disturbances and additional settings that can be enabled at the bottom of the file.

See below for what the meshcat visualization should like.
![Meshcat visualization](media/meshcat-cf.png)