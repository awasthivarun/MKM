# Raw Data Policy
Files in this directory are treated as raw experimental data.
Files under `data/raw/` should not be modified by preprocessing or modeling code. Corrections, exclusions, or transformations should not be applied in place.
Any cleaned, transformed, interpolated, truncated, averaged, or derived datasets should be generated from these files and stored outside `data/raw/`.