from subprocess import run
from pathlib import Path
import os
import shutil

DATA_PATH = Path(__file__).parent / 'data'
NOTEBOOKS = Path(__file__).parent.parent.parent / 'notebooks'

def test_that_filecal_runs_through(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")

    fn = tmp / "zen.2459847.51336.sum.html"
    
    notebook = NOTEBOOKS/'file_calibration.ipynb'
    
    # Copy the SUMFILE to a tmp directory, so the outputs go to the tmp directory.
    sumfile = DATA_PATH/'zen.2459847.51336.sum.downsampled.uvh5'
    difffile = DATA_PATH / sumfile.name.replace(".sum.", ".diff.")
    shutil.copy(sumfile, tmp / sumfile.name)
    shutil.copy(difffile, tmp / difffile.name)
    
    sumfile = tmp / sumfile.name
    
    result = run(
    [
        "jupyter", "nbconvert",
        "--to", "html",
        "--execute",
        "--output", str(fn),
        str(notebook)
    ],
    env={
        **os.environ,
        **{
            "SUM_FILE": str(sumfile),
            "AUTO_SLOPE_GOOD_LOW": "-1.0",  # Set these wider than usual, because we're
            "AUTO_SLOPE_GOOD_HIGH": "1.0",  # using a subset of frequency.
            "OC_MIN_DIM_SIZE": "0",         # Default is 8, but we don't have that many redundant bls in our small set.
        }
    }
    )

    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0

    for ext in [".ant_class.csv", ".ant_metrics.hdf5",".omni_vis.uvh5", ".omni.calfits"]:
        assert sumfile.with_suffix(ext).exists()
