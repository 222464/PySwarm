# PySwarm

Python bindings to [Swarm](https://github.com/222464/Swarm)
## Requirements

An install of [Swarm](https://github.com/222464/Swarm) is required before installing the bindings.

This binding requires an installation of [SWIG](http://www.swig.org/) v3+

#### [SWIG](http://www.swig.org/)

- Linux requires SWIG installed via, for example ```sudo apt-get install swig3.0``` command (or via ```yum```).
- Windows requires installation of SWIG (v3). With the SourceForge Zip expanded, and the PATH environment variable updating to include the SWIG installation binary directory (for example `C:\Program Files (x86)\swigwin-3.0.8`).
- Mac OSX can use Homebrew to install the latest SWIG (for example, see .travis/install_swig.sh Bash script).

## Installation

The following example can be used to build the Python package:

> python3 setup.py install --user

## Importing and Setup

The PySwarm module can be imported using:

```python
import pyswarm
```

See the examples directory for usage.

## License

MIT license. See [LICENSE.md](./LICENSE.md) for more information.
