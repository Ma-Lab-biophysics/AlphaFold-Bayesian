# Third-party notices

This package includes components adapted from the third-party BioEn package:
https://github.com/bio-phys/BioEn

## BioEn-adapted optimization components

The local `multi-d` optimization backend includes BioEn-derived implementation logic for log-weights optimization. The mathematical objective and implementation structure are adapted from the relevant BioEn components, including:

- `bioen/optimize/log_weights.py`
- `bioen/optimize/common.py`

The backend is implemented locally in this package and does not import or execute the upstream BioEn Python modules at runtime.

## License compatibility

BioEn is distributed under the GNU General Public License v3.0. This package is therefore distributed as GPL-3.0-or-later.
