"""Tests of the single-baseline pspect notebook."""

from click.testing import CliRunner
from hera_notebook_templates import NOTEBOOKS, cli
from pathlib import Path
import pytest

DATA_PATH = Path(__file__).parent / "data"

def test_single_bl_pspec(tmp_path):
    if not (DATA_PATH / "NF_HERA_Vivaldi_efield_beam_healpix.fits").exists():    
        pytest.xfail("We don't have the beam file in the repo because it's too big")
    
    runner = CliRunner()
    out = runner.invoke(
        cli.run, 
        [
            "single_baseline_postprocessing_and_pspec",
            "--SINGLE-BL-FILE", str(DATA_PATH / "example_lstbinned_baseline.0_1.uvh5"),
            "--OUT-PSPEC-FILE", str(tmp_path / "out_pspec.0_1.pspec.h5"),
            "--OUT-TAVG-PSPEC-FILE", str(tmp_path / "out_pspec.0_1.tavg.pspec.h5"),
            "--BAND-STR", '108.1~115.8,117.3~124.4,125.5~135.9,138.8~147.1,150.3~161.4',
            "--FR-SPECTRA-FILE", str(DATA_PATH / "frf_spectra_cache_downsampled.h5"),
            "--USE-BAND-AVG-NSAMPLES",
            "--EFIELD-HEALPIX-BEAM-FILE", str(DATA_PATH / "NF_HERA_Vivaldi_efield_beam_healpix.fits")
        ]
    )
    
    if out.exit_code != 0:
        print(out.stdout)
        raise out.exception
    
    assert (tmp_path / 'out_pspec.0_1.pspec.h5').exists()
    assert (tmp_path / 'out_pspec.0_1.tavg.pspec.h5').exists()
    