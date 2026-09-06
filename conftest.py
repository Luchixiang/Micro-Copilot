import pytest
import os, tempfile, zipfile
from pathlib import Path
from urllib.error import URLError

TEST_MODEL_DIR = Path(tempfile.gettempdir()).joinpath('cellquant-test-models')
TEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('CELLPOSE_LOCAL_MODELS_PATH', os.fspath(TEST_MODEL_DIR))

from cellquant import utils

@pytest.fixture(scope='session')
def image_names():
    image_names = [
        "gray_2D.png", "rgb_2D.png", "rgb_2D_tif.tif", "gray_3D.tif", "rgb_3D.tif",
        "segment_80x224x448_input.tiff", "segment_80x224x448_expected.tiff"
    ]
    return image_names


@pytest.fixture(scope='session')
def data_dir(image_names, tmp_path_factory):
    configured_dir = os.environ.get('CELLQUANT_TEST_DATA_DIR')
    if configured_dir:
        return Path(configured_dir)

    download_dir = tmp_path_factory.mktemp('cellquant-data')
    archive_path = download_dir.joinpath('data.zip')
    try:
        utils.download_url_to_file('https://osf.io/download/s52q3/', archive_path)
    except URLError as exc:
        pytest.skip(f'official Cellpose test data are unavailable: {exc}')
    with zipfile.ZipFile(archive_path, 'r') as archive:
        archive.extractall(download_dir)
    return download_dir.joinpath('data')
