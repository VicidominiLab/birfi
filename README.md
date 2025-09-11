# BIRFI


[![License](https://img.shields.io/pypi/l/birfi.svg?color=green)](https://github.com/VicidominiLab/birfi/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/birfi.svg?color=green)](https://pypi.org/project/birfi/)
[![Python Version](https://img.shields.io/pypi/pyversions/birfi.svg?color=green)](https://python.org)

Blind instrument response function identification from fluorescence decays.
This is a Python re-implementation of the algorithm described in:
[Adrián Gómez-Sánchez et al., _Blind instrument response function identification from fluorescence decays_,
Biophysical Reports, 2024](https://doi.org/10.1016/j.bpr.2024.100155).

It works with single-channel and multi-channel (e.g. ISM) datasets.


## Installation

You can install `birfi` via [pip] directly from GitHub:

    pip install git+https://github.com/VicidominiLab/birfi

Currently, we reccomend cloning the repository and installing the package in developer mode.
Go to the repository folder and run:

    pip install -e .

It requires the following Python packages

    numpy
    matplotlib
    torch

## License

Distributed under the terms of the [GNU GPL v3.0] license.
"birfi" is free and open source software


[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt

[file an issue]: https://github.com/VicidominiLab/birfi/issues

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/project/birfi/
