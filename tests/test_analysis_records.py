import cv2
import numpy as np
import pandas as pd

from cellquant.analysis import PunctaDetectorProcess


def make_analyzer():
    analyzer = PunctaDetectorProcess(None, None, None, None)
    analyzer.time_point = -1
    analyzer.chan0 = 0
    return analyzer


def test_two_channel_colocalization_is_recorded_both_directions(tmp_path):
    analyzer = make_analyzer()
    image_path = tmp_path / 'sample.ome.tif'
    image_path.touch()

    channel_1_blobs = [np.array([[10.0, 10.0, 2.0], [40.0, 40.0, 2.0]])]
    channel_2_blobs = [np.array([[11.0, 10.0, 2.0]])]
    channel_1_intensities = [[100.0, 200.0]]
    channel_2_intensities = [[300.0]]

    co_12 = analyzer.calculate_distance(channel_2_blobs, channel_1_blobs, channel_1_intensities)
    analyzer.write_co_localization(
        0, 1, str(image_path), channel_1_blobs, channel_2_blobs, *co_12,
        distance_threshold=4)
    co_21 = analyzer.calculate_distance(channel_1_blobs, channel_2_blobs, channel_2_intensities)
    analyzer.write_co_localization(
        1, 0, str(image_path), channel_2_blobs, channel_1_blobs, *co_21,
        distance_threshold=4)

    puncta = pd.read_csv(tmp_path / 'sample_structure_chan1_cell1_time-1.csv')
    assert puncta['in chan2'].tolist() == [1.0, 0.0]

    summary_path = tmp_path / 'sample_colocalization.csv'
    summary = pd.read_csv(summary_path)
    assert len(summary) == 2
    first_direction = summary[(summary.source_channel == 1) & (summary.target_channel == 2)].iloc[0]
    assert first_direction.source_puncta_count == 2
    assert first_direction.colocalized_puncta_count == 1
    assert first_direction.colocalization_ratio == 0.5

    # Re-running a direction updates its rows instead of creating duplicates.
    analyzer.write_co_localization(
        0, 1, str(image_path), channel_1_blobs, channel_2_blobs, *co_12,
        distance_threshold=4)
    assert len(pd.read_csv(summary_path)) == 2


def test_analysis_plan_adds_directional_colocalization_for_selected_channels():
    analyzer = make_analyzer()
    plan = analyzer.get_analysis_plan('', [0, 1, -1], -1)
    pairs = {(item['source_channel'], item['target_channel']) for item in plan['colocalizations']}
    assert pairs == {(0, 1), (1, 0)}


def test_cell_segmentation_overlay_is_saved_beside_input(tmp_path):
    analyzer = make_analyzer()
    image_path = tmp_path / 'sample.tif'
    image = np.zeros((32, 32), dtype=np.uint16)
    image[8:24, 8:24] = 1000
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1

    analyzer.save_analysis_cell_mask(mask, image, str(image_path))

    overlay_path = tmp_path / 'sample_analysis_cell_seg.png'
    saved_mask_path = tmp_path / 'sample_analysis_cell_mask.npy'
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    assert overlay is not None
    assert overlay.shape == (32, 32, 3)
    np.testing.assert_array_equal(np.load(saved_mask_path), mask)
