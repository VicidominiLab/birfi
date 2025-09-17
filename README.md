# BIRFI


[![License](https://img.shields.io/pypi/l/birfi.svg?color=green)](https://github.com/VicidominiLab/birfi/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/birfi.svg?color=green)](https://pypi.org/project/birfi/)
[![Python Version](https://img.shields.io/pypi/pyversions/birfi.svg?color=green)](https://python.org)

Blind Instrument Response Function Identification (BIRFI) from fluorescence decays.
This is a Python re-implementation of the algorithm described in:
[Adrián Gómez-Sánchez et al., _Blind instrument response function identification from fluorescence decays_,
Biophysical Reports, 2024](https://doi.org/10.1016/j.bpr.2024.100155).

It works with single-channel and multi-channel (e.g. ISM) datasets.


## Installation

You can install the stable version from [PyPI]:

    pip install git+https://github.com/VicidominiLab/birfi

or the latest version directly from GitHub:

    pip install git+https://github.com/VicidominiLab/birfi

It requires the following Python packages

    numpy
    scipy
    matplotlib
    torch

## Documentation

The algorithm calculates the IRFs from a single-channel or multi-channel fluorescence decay dataset, assuming that
the fluorescence decays are mono-exponential, and they share the same lifetime. The dataset should be in the shape of
(n_time_bins,) or (n_time_bins, n_channels). The algorithm is sensitive to noise, so we recommend acquiring 
calibration data with the highest possible signal-to-noise ratio. In case this is not possible, we provide a simple
regularization tool to minimize noise overfitting.

You can find examples of usage here:

https://github.com/VicidominiLab/birfi/tree/main/demo


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
