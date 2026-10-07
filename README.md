# HERA_HONS
Repository dedicated to the HERA 21 cm analysis pipeline configuration project in the Honours year for Bernard Weich.

## Required data files

The following large data files are required to run the project but are not
stored in the Git repository because of their size:

- `src/HONS/NF_HERA_Vivaldi_efield_beam_healpix.fits`
  - HERA Vivaldi E-field beam model.
- `src/spectra_cache_hera_core.h5`
  - HERA spectra cache required by the analysis pipeline.

These files must be obtained separately and placed at the paths above before
running the analysis.