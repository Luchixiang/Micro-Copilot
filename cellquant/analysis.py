import cv2
import numpy as np
import scipy.ndimage
from scipy.ndimage import distance_transform_edt
import csv
import warnings

from skimage import filters
from skimage.feature import blob_log
from skimage.feature.blob import _prune_blobs
from skimage.draw import disk
from scipy.spatial import KDTree
from cellquant.utils import *
from cellquant.puncta_classifier.classifier import PunctaClassifier

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from multiprocessing import Process, Queue
from skimage.color import label2rgb
from skimage.feature import blob_log
from skimage.measure import regionprops
from skimage.morphology import closing
from skimage.filters import threshold_otsu, sobel
import math
import pandas as pd
import time
import os
import re
import json
import hashlib
import importlib.util
import subprocess
import difflib

import cellquant.models as cellpose_models
from cellquant import core as cellpose_core
import logging




logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _image_stem(file):
    """Return an image name without common microscopy extensions."""
    name = os.path.basename(os.fspath(file))
    lower_name = name.lower()
    for suffix in ('.ome.tiff', '.ome.tif', '.tiff', '.tif', '.png', '.jpg', '.jpeg'):
        if lower_name.endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]

PYCRO_STATUS = 1

try:
    import pycromanager
except:
    PYCRO_STATUS = 0


class AnalysisInstructionParser:
    """Parse GUI analysis language into an executable analysis plan."""

    CHANNEL_WORDS = {
        'first': 1, '1st': 1, 'one': 1,
        'second': 2, '2nd': 2, 'two': 2,
        'third': 3, '3rd': 3, 'three': 3,
        'fourth': 4, '4th': 4, 'four': 4,
    }

    LYSOSOME_TERMS = ('lysosome', 'lysosomes', 'lyso', 'lysotracker', 'lysotracker')
    NUCLEUS_TERMS = ('nucleus', 'nuclei', 'nuclear', 'dapi')
    CELL_TERMS = ('cell', 'cells', 'cytoplasm')
    ACTION_TERMS = ('detect', 'segment', 'instance', 'analyze', 'analyse', 'measure', 'calculate')
    FUZZY_TERMS = LYSOSOME_TERMS + NUCLEUS_TERMS + CELL_TERMS + ACTION_TERMS + (
        'channel', 'chan', 'colocalization', 'colocalisation', 'localization', 'localisation', 'ratio')


    def parse(self, instruction, fallback_channels=None, fallback_nuclear_channel=-1):
        instruction = self.normalize_instruction(instruction)

        plan = {
            'instruction': instruction,
            'detections': [],
            'segmentations': [],
            'colocalizations': [],
            'nuclear_channel': fallback_nuclear_channel,
        }
        if not instruction:
            return plan

        normalized = instruction.lower().replace('co localization', 'colocalization')
        clauses = [clause.strip() for clause in re.split(r'[;\n]+|\.\s+|\bthen\b', normalized) if clause.strip()]
        for clause in clauses:
            channels = self._channels_in_text(clause)
            if not channels:
                continue

            if self._contains_any(clause, self.NUCLEUS_TERMS) and ('segment' in clause or 'detect' in clause):
                for chan in channels:
                    self._add_unique(plan['segmentations'], {
                        'object': 'nucleus', 'channel': chan, 'method': 'llm_custom'
                    })
                plan['nuclear_channel'] = channels[0]
                continue

            if self._contains_any(clause, self.CELL_TERMS) and ('segment' in clause or 'detect' in clause):
                for chan in channels:
                    self._add_unique(plan.setdefault('cell_segmentations', []), {
                        'object': 'cell', 'channel': chan, 'method': 'cellpose'
                    })
                continue

            if self._contains_any(clause, self.LYSOSOME_TERMS) and ('detect' in clause or 'analyze' in clause or 'measure' in clause):

                for chan in channels:
                    self._add_unique(plan['detections'], {
                        'object': 'lysosome', 'channel': chan, 'tool': 'cellquant_lysosome'
                    })

            if ('segment' in clause or 'threshold' in clause or 'detect' in clause) and not self._contains_any(clause, self.LYSOSOME_TERMS):
                object_name = self._object_name(clause)
                for chan in channels:
                    self._add_unique(plan['segmentations'], {
                        'object': object_name, 'channel': chan, 'method': 'llm_custom'
                    })

            if 'colocaliz' in clause or 'co-localiz' in clause or 'localization ratio' in clause or 'localisation ratio' in clause:

                pairs = self._colocalization_pairs(clause, channels)
                for first_chan, second_chan in pairs:
                    self._add_unique(plan['colocalizations'], {
                        'source_channel': first_chan,
                        'target_channel': second_chan,
                        'metric': 'ratio'
                    })

        if not plan['detections'] and not plan['segmentations'] and not plan.get('cell_segmentations') and fallback_channels:

            for chan in fallback_channels:
                if chan is not None and chan >= 0:
                    self._add_unique(plan['detections'], {
                        'object': 'lysosome', 'channel': chan, 'tool': 'cellquant_lysosome'
                    })
        return plan

    def normalize_instruction(self, instruction):
        text = (instruction or '').strip().lower()
        text = re.sub(r'\bco\s*[-_]?\s*locali[sz]\w*\b', 'colocalization', text)
        text = re.sub(r'\blocali[sz]\w*\s+ratio\b', 'localization ratio', text)
        text = re.sub(r'\bch\s*(\d+)\b', r'channel \1', text)
        text = re.sub(r'\bc\s*(\d+)\b', r'channel \1', text)
        text = re.sub(r'\bchan\s*(\d+)\b', r'channel \1', text)
        text = re.sub(r'\blyso\b', 'lysosome', text)
        text = re.sub(r'\blysotrack(?:er)?\b', 'lysotracker', text)

        def fix_word(match):
            word = match.group(0)
            if len(word) < 4 or word in self.FUZZY_TERMS:
                return word
            candidates = difflib.get_close_matches(word, self.FUZZY_TERMS, n=1, cutoff=0.78)
            return candidates[0] if candidates else word

        return re.sub(r'\b[a-zA-Z][a-zA-Z_-]*\b', fix_word, text)

    def _channels_in_text(self, text):
        channels = []

        for match in re.finditer(r'\b(?:channel|chan|ch|c)\s*[-_ ]?(\d+)\b', text):
            self._append_channel(channels, int(match.group(1)) - 1)
        if 'channel' in text or 'chan' in text or re.search(r'\bc\s*\d+\b', text):
            for word, channel_number in self.CHANNEL_WORDS.items():
                if re.search(rf'\b{re.escape(word)}\b', text):
                    self._append_channel(channels, channel_number - 1)
        return channels

    def _colocalization_pairs(self, clause, channels):
        pairs = []
        numeric_pair_pattern = r'\b(?:channel|chan|ch|c)\s*[-_ ]?(\d+)\b\s+in\s+\b(?:channel|chan|ch|c)\s*[-_ ]?(\d+)\b'
        for match in re.finditer(numeric_pair_pattern, clause):
            first_chan = int(match.group(1)) - 1
            second_chan = int(match.group(2)) - 1
            if first_chan >= 0 and second_chan >= 0:
                pairs.append((first_chan, second_chan))
        if not pairs and len(channels) >= 2:
            pairs.append((channels[0], channels[1]))
        return pairs

    def _object_name(self, clause):
        if self._contains_any(clause, self.NUCLEUS_TERMS):
            return 'nucleus'
        match = re.search(r'\b(?:segment|detect|measure|analyze)\s+(.+?)\s+(?:in|on|from)\b', clause)
        if match:
            name = match.group(1).strip()
            name = re.sub(r'\b(the|a|an|using|with|by|intensity|based|otsu|threshold)\b', '', name).strip()
            name = re.sub(r'\s+', '_', name)
            return name or 'structure'
        return 'structure'

    def _contains_any(self, text, terms):
        return any(term in text for term in terms)

    def _append_channel(self, channels, channel):
        if 0 <= channel <= 3 and channel not in channels:
            channels.append(channel)

    def _add_unique(self, collection, item):
        if item not in collection:
            collection.append(item)


class PunctaDetectorProcess(Process):

    organelle_chans = ['C1', 'C1 in C2', 'C1 in C3', 'C1 in C4', 'C2', 'C2 in C1', 'C2 in C3', 'C2 in C4',
                       'C3', 'C3 in C1', 'C3 in C2', 'C3 in C4', 'C4', 'C4 in C1', 'C4 in C2', 'C4 in C3']
    organelle_properties = ['structure radius', 'structure nuclear distance', 'structure intensity', 'structure number',
                            'co-localization ratio']
    metrics = ['Name']
    for i in organelle_chans:
        for j in organelle_properties:
            if 'in' not in i and j == 'co-localization ratio':  # not co, no need to ratip
                continue
            if 'in' in i and (
                    j == 'structure nuclear distance' or j == 'structure radius'):  # co, no need for dis and radius
                continue
            metrics.append(i + ' ' + j)

    def __init__(self, queue, done_event, plot_queue, result_queue):
        super().__init__()
        self.queue = queue
        self.done_event = done_event
        self.plot_queue = plot_queue
        self.result_queue = result_queue
        self.model_path = None
        self.instruction_parser = AnalysisInstructionParser()
        self.resize_scale = 4
        self.cell_min_size = 200 * 200
        self.default_cellpose_model = None
        self.default_cellpose_model_key = None
        self.analysis_plan_cache = {}





    def get_center_cells(self, masks):
        if masks is None:
            return []
        masks = np.asarray(masks)
        if masks.size == 0 or masks.ndim < 2:
            return []
        height, width = masks.shape[-2:]
        center_cells = []
        max_label = int(np.max(masks)) if masks.size > 0 else 0
        for cell in range(1, max_label + 1):
            cell_mask = (masks == cell)

            upper = np.sum(cell_mask[:10, :])
            lower = np.sum(cell_mask[-10:, :])
            left = np.sum(cell_mask[:, :10])
            right = np.sum(cell_mask[:, -10:])
            # totally in the center
            if upper <= 300 * 10 and lower <= 300 * 10 and left <= 300 * 10 and right <= 300 * 10:
                center_cells.append(cell)
                continue
            y_indices, x_indices = np.where(cell_mask != 0)
            centeroid = (np.mean(y_indices), np.mean(x_indices))
            if abs(centeroid[1] - width // 2) < 300 and abs(centeroid[0] - height // 2) < 300:
                center_cells.append(cell)
                continue
        return center_cells

    def puncta_detection(self, img, masks, file, channel, threshold, center_cells, useDL=True, nuclear_image=None):
        if img is None:
            warnings.warn('puncta_detection received no image')
            return [], [], [], []
        if masks is None:
            masks = np.ones_like(img, dtype=np.uint8)
        if center_cells is None:
            center_cells = self.get_center_cells(masks)
            if len(center_cells) == 0 and np.max(masks) >= 1:
                center_cells = [1]
        if len(center_cells) == 0:
            return [], [], [], []

        if threshold is None:
            threshold = 0.2
        input_img = img.copy()
        final_draw_image = img.copy()

        final_draw_image = change_to_8bit(final_draw_image)
        final_draw_image = cv2.cvtColor(final_draw_image, cv2.COLOR_GRAY2BGR)
        segmentation_image = np.zeros_like(img).astype(np.uint8)
        img = remove_background(img)
        all_blobs = []
        all_nuclear_distances = []
        all_intensities = []
        all_areas = []
        if center_cells is None and self.time_point > 0:
            center_cells = self.get_center_cells(masks)
        if center_cells is None:
            center_cells = []
        if nuclear_image is not None:

            nuclear_image = filters.gaussian(nuclear_image.copy(), sigma=5)
        # img = filters.gaussian(img.copy(), sigma=1)
        for num_cell in center_cells:
            cell_blobs = []
            current_cell_mask = (masks == num_cell)
            if np.sum(current_cell_mask) <= 200 * 100: # some cells will disapear.
                all_blobs.append(cell_blobs)
                nuclear_distances = [None for _ in range(len(cell_blobs))]
                all_nuclear_distances.append(nuclear_distances)
                continue
            cell_image_noseg = filters.gaussian(img.copy(), sigma=1)
            cell_image_noseg = normalize_mask(cell_image_noseg, current_cell_mask)
            if nuclear_image is not None:
                nuclear_image_cell = normalize_mask(nuclear_image, current_cell_mask)
                nuclear_image_cell = self.nuclear_segmentation(nuclear_image_cell, file)
                nuclear_image_cell[current_cell_mask != 1] = 0
            draw_image = cell_image_noseg.copy()
            draw_image = cv2.cvtColor(draw_image, cv2.COLOR_GRAY2BGR)
            origin_image = draw_image.copy()
            # starty = 0
            # startx = 0
            startx, starty, cropw, croph = self.crop_mask(cell_image_noseg, current_cell_mask)
            cell_image_noseg_crop = cell_image_noseg[starty:starty + croph, startx:startx + cropw]
            # cell_image_noseg_w = change_to_8bit(cell_image_noseg)
            # cv2.imwrite(os.path.join(os.path.dirname(file), file.replace('.tif', f'puncta_classifier_{channel}_numcell{num_cell}.png')), cell_image_noseg_w)
            start_time = time.time()
            blobs_candidates = blob_log(cell_image_noseg_crop, min_sigma=1, max_sigma=10, num_sigma=10,
                                        threshold=threshold,
                                        overlap=0.3)
            blobs_candidates = [(blob[0] + starty, blob[1] + startx, blob[2]) for blob in blobs_candidates if
                                masks[int(blob[0] + starty), int(blob[1] + startx)] == num_cell and blob[2] >= 3]
            logger.info(f"'candidate proposal time', {time.time() - start_time}")
            for blob in blobs_candidates:
                y, x, r = blob
                cv2.circle(draw_image, (int(x), int(y)), int(r * 1.4), color=(255, 0, 0), thickness=2)

            start_time = time.time()
            if useDL and getattr(self, 'puncta_classifier', None) is not None:

                predictions = self.puncta_classifier.pred(blobs_candidates,
                                                          origin_image.copy(),
                                                          draw_image.copy())  # do the neural network classification, remove the FP
                logger.info(f"'candidate classification time', {time.time() - start_time}")
            else:
                predictions = [0 for _ in range(len(blobs_candidates))]
            for i in range(len(blobs_candidates)):
                blob = blobs_candidates[i]
                y, x, r = blob
                if predictions[i] == 0:
                    cv2.circle(final_draw_image, (int(x), int(y)), int(r * 1.4), color=(255, 0, 0),
                               thickness=2)
                    cell_blobs.append([y, x, r])
                elif predictions[i] == 1:
                    cv2.circle(final_draw_image, (int(x), int(y)), int(r * 1.4), color=(0, 255, 0),
                               thickness=2)
                elif predictions[i] == 2:
                    cv2.circle(final_draw_image, (int(x), int(y)), int(r * 1.4), color=(0, 0, 255),
                               thickness=2)
                    # cell_blobs.append([y, x, r])
            merged_circles = self.merge_circles(
                [blobs_candidates[i] for i, pred in enumerate(predictions) if pred == 2], threshold=20)
            for y, x, r in merged_circles:
                cv2.circle(final_draw_image, (int(x), int(y)), int(r * 1.4), color=(0, 255, 255),
                           thickness=2)
                cell_blobs.append([y, x, r])
            if len(cell_blobs) > 0:
                cell_blobs = _prune_blobs(np.array(cell_blobs), 0.3, sigma_dim=1)

            nuclear_distances = [None for _ in range(len(cell_blobs))]
            if nuclear_image is not None and len(cell_blobs) > 0:
                if np.sum(nuclear_image_cell) != 0:
                    nuclear_distances = self.calculate_nuclear_dis(cell_blobs, nuclear_image_cell, current_cell_mask, file)
            intensities = []
            areas = []
            start_time = time.time()
            primary_segmentation = np.zeros_like(img).astype(np.uint8)
            for idx, blob in enumerate(cell_blobs):
                y, x, r = blob
                rr, cc = disk((int(y), int(x)), int(r * 1.4), shape=img.shape)
                primary_segmentation[rr, cc] = idx + 1
            for idx, blob in enumerate(cell_blobs):
                rr, cc = self.puncta_segmentation(cell_image_noseg, blob, segmentation_image, primary_segmentation)
                segment = input_img[rr, cc]
                intensity = np.mean(segment)
                area = segment.shape[0]
                intensities.append(intensity)
                areas.append(area)
            logger.info(f"'puncta segmentation time', {time.time() - start_time}")
            all_blobs.append(cell_blobs)
            all_nuclear_distances.append(nuclear_distances)
            all_intensities.append(intensities)
            all_areas.append(areas)
        segmentation_image = label2rgb(segmentation_image, (final_draw_image.copy()), bg_label=0) * 255
        image_stem = _image_stem(file)
        cv2.imwrite(get_unique_filename(os.path.dirname(file),
                                        f'{image_stem}_structure_chan{channel + 1}_seg.png'),
                    segmentation_image.astype(np.uint8))
        image_name = get_unique_filename(os.path.dirname(file),
                                         f'{image_stem}_structure_chan{channel + 1}.png')
        cv2.imwrite(image_name, final_draw_image)
        return all_blobs, all_nuclear_distances, all_intensities, all_areas

    def cell_segmentation_track(self, record, prev_mask, img,
                                file):  # cell_channel, use_gpu, model_path, do_3D, stitch_threshold, resize_scale, cell_min_size, channels, flow_threshold, cellprob_threshold, diameter, niter,normalize_params
        model_path = record['model_path']
        channels = record['cell_channel']
        if img.ndim > 2:
            img = img[channels, :, :]
        if model_path != self.model_path:
            self.model_path = model_path
            self.segmentation_model = cellpose_models.CellposeModel(gpu=record['use_gpu'],
                                                                    pretrained_model=model_path)
        self.flows = [[], [], []]
        do_3D = record['do_3D']
        stitch_threshold = record['stitch_threshold']
        do_3D = False if stitch_threshold > 0. else do_3D
        w, h = img.shape
        self.resize_scale = record['resize_scale']
        image = cv2.resize(img, None, fx=1. / self.resize_scale, fy=1. / self.resize_scale,
                           interpolation=cv2.INTER_LINEAR)
        min_size = record['cell_min_size']
        min_size = min_size // (self.resize_scale ** 2)
        channels = record['channels']
        flow_threshold = record['flow_threshold']
        cellprob_threshold = record['cellprob_threshold']
        # diameter = self.ui.diameter / self.resize_scale
        diameter = record['diameter']
        niter = max(0, int(record['niter']))
        niter = None if niter == 0 else niter
        normalize_params = record['normalize_params']
        masks, flows = self.segmentation_model.eval(
            image, channels=channels, diameter=diameter,
            cellprob_threshold=cellprob_threshold, min_size=min_size,
            flow_threshold=flow_threshold, do_3D=do_3D, niter=niter,
            normalize=normalize_params, stitch_threshold=stitch_threshold,
        )[:2]
        masks = masks.astype(np.uint8)
        current_mask = cv2.resize(masks, None, fx=self.resize_scale, fy=self.resize_scale,
                                  interpolation=cv2.INTER_NEAREST)
        masks = self.cell_track(prev_mask, current_mask)
        colored_mask = label2rgb(masks, change_to_8bit(img.copy()), bg_label=0) * 255
        if file is not None:
            cv2.imwrite(os.path.join(os.path.dirname(file),
                                     f'{_image_stem(file)}_t{self.time_point}_seg.png'),
                        colored_mask.astype(np.uint8))
        return masks

    def cell_track(self, prev_mask, current_mask):
        """
        correlate cells from the last timepoint to the current timepoint.
        """

        def _compute_features(props):
            """ Extracts relevant features including position and shape properties. """
            return np.array([(prop.centroid[0],  # Y-coordinate
                              prop.centroid[1],  # X-coordinate
                              prop.area,
                              prop.perimeter,
                              prop.eccentricity,
                              ) for prop in props])

        # if len(np.unique(prev_mask)) != len(np.unique(current_mask)): # when two c
        #     return prev_mask
        if len(np.unique(prev_mask)) == 1 or len(np.unique(current_mask)) == 1:
            return prev_mask
        props_prev = regionprops(prev_mask)
        props_current = regionprops(current_mask)
        features_prev = _compute_features(props_prev)
        features_current = _compute_features(props_current)
        max_features = np.max(np.vstack((features_prev, features_current)), axis=0)
        features_prev /= max_features
        features_current /= max_features

        weights = np.array([3, 3, 1, 1, 1])  # Increase weights for Y and X centroids
        # Create weighted distance matrix using custom lambda function
        distance_matrix = cdist(features_prev, features_current, lambda u, v: np.dot(weights, np.abs(u - v)))
        if np.isnan(np.sum(distance_matrix)):
            return prev_mask
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        # print(distance_matrix, self.time_point)
        dissimilarity_threshold = 0.5  # todo: Adjust this value
        # Filter matches based on the threshold
        valid_matches = [(row, col) for row, col in zip(row_ind, col_ind) if
                         distance_matrix[row, col] <= dissimilarity_threshold]
        related_mask = np.zeros_like(current_mask)
        for row, col in valid_matches:
            # print(f"Cell with label {row + 1} in mask A matches with Cell with label {col + 1} in mask B")
            # print(np.sum(current_mask == (col+1)))
            related_mask[current_mask == (col + 1)] = row + 1

        # 1. one cell in previous not show in current
        for cell in range(1, np.max(prev_mask) + 1):
            if np.sum(related_mask == cell) < 100 * 100:
                related_mask[(prev_mask == cell) & (related_mask == 0)] = cell
        # for cell in range(1, np.max(current_mask)):
        #     if np.sum(related_mask == cell) < 100 * 100:

        return related_mask

    def crop_mask(self, img, mask):
        mask = mask.astype(np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours) == 0:
            return 0, 0, img.shape[1], img.shape[0]
        x, y, w, h = cv2.boundingRect(contours[0])

        x -= 10
        y -= 10
        w += 20
        h += 20
        x = max(x, 0)
        y = max(y, 0)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        return x, y, w, h

    def norm(self, image):
        img_min = image.min()
        img_max = image.max()
        image -= img_min
        if img_max > img_min + 1e-3:
            image /= (img_max - img_min)
        image *= 255
        return image.astype(np.uint16)

    def get_analysis_plan(self, instruction, detection_channels, nuclear_channel):
        detection_channels = list(detection_channels or [])
        nuclear_channel = -1 if nuclear_channel is None else nuclear_channel
        cache_key = (instruction or '', tuple(detection_channels or []), nuclear_channel)

        if cache_key in self.analysis_plan_cache:
            return json.loads(json.dumps(self.analysis_plan_cache[cache_key]))

        if instruction:
            try:
                plan = self.get_analysis_plan_from_llm(instruction, detection_channels, nuclear_channel)
                self.analysis_plan_cache[cache_key] = plan
                return json.loads(json.dumps(plan))
            except Exception as e:
                feedback = self.format_llm_feedback(e, context='instruction parsing')
                warnings.warn(f'{feedback} Using local parser fallback.')
                logger.warning(f'{feedback} Using local parser fallback.')

        plan = self.instruction_parser.parse(instruction, detection_channels, nuclear_channel)

        if not instruction:
            for chan in detection_channels:
                if chan is not None and chan >= 0:
                    plan['detections'].append({
                        'object': 'lysosome',
                        'channel': chan,
                        'tool': 'cellquant_lysosome'
                    })
            for first_chan in detection_channels:
                for second_chan in detection_channels:
                    if first_chan is not None and second_chan is not None and first_chan >= 0 and second_chan >= 0 and first_chan != second_chan:
                        plan['colocalizations'].append({
                            'source_channel': first_chan,
                            'target_channel': second_chan,
                            'metric': 'ratio'
                        })
        self.analysis_plan_cache[cache_key] = plan
        return json.loads(json.dumps(plan))

    def get_analysis_plan_from_llm(self, instruction, detection_channels, nuclear_channel):
        prompt = self.build_instruction_parse_prompt(instruction, detection_channels, nuclear_channel)
        response = self.call_llm(
            prompt,
            system_content='You convert microscopy analysis requests into strict JSON plans. Return only JSON.'
        )
        data = self.extract_json_block(response)
        return self.sanitize_llm_plan(data, instruction, fallback_nuclear_channel=nuclear_channel)

    def build_instruction_parse_prompt(self, instruction, detection_channels, nuclear_channel):
        available_channels = [chan + 1 for chan in detection_channels if chan is not None and chan >= 0]
        return f"""
Parse this microscopy analysis instruction into JSON. Correct typos and abbreviations first.
Instruction: {instruction}
Fallback detection channels if the user is vague: {available_channels}
Fallback nuclear channel, one-based if >=1 else none: {nuclear_channel + 1 if nuclear_channel is not None and nuclear_channel >= 0 else None}

Return ONLY JSON with this schema:
{{
  "normalized_instruction": "corrected instruction text",
  "detections": [{{"object": "lysosome", "channel": 1, "tool": "cellquant_lysosome"}}],
  "segmentations": [{{"object": "nucleus", "channel": 1, "method": "llm_custom"}}],
  "cell_segmentations": [{{"object": "cell", "channel": 1, "method": "cellpose"}}],
  "colocalizations": [{{"source_channel": 1, "target_channel": 2, "metric": "ratio"}}],
  "nuclear_channel": null
}}

Rules:
- Use ONE-BASED channel numbers in JSON.
- Lysosome/lyso/lysotracker-like words mean object lysosome and tool cellquant_lysosome.
- Cell/cytoplasm instance segmentation means cell_segmentations with method cellpose.
- Nucleus/nuclear/DAPI segmentation means segmentations with method llm_custom and nuclear_channel.
- Other structures use segmentations with method llm_custom.
- Colocalization/localization ratio/source in target means source_channel in target_channel.
- Empty arrays are allowed. Do not add explanations.
""".strip()

    def extract_json_block(self, text):
        match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
        json_text = match.group(1).strip() if match else text.strip()
        start = json_text.find('{')
        end = json_text.rfind('}')
        if start >= 0 and end >= start:
            json_text = json_text[start:end + 1]
        return json.loads(json_text)

    def sanitize_llm_plan(self, data, original_instruction, fallback_nuclear_channel=-1):
        plan = {
            'instruction': data.get('normalized_instruction') or original_instruction,
            'detections': [],
            'segmentations': [],
            'colocalizations': [],
            'nuclear_channel': fallback_nuclear_channel,
        }
        for item in data.get('detections', []) or []:
            channel = self.one_based_to_zero_based(item.get('channel'))
            if channel is not None:
                obj = str(item.get('object', 'structure')).lower()
                tool = 'cellquant_lysosome' if 'lyso' in obj else item.get('tool', 'cellquant_lysosome')
                plan['detections'].append({'object': 'lysosome' if tool == 'cellquant_lysosome' else obj,
                                           'channel': channel, 'tool': tool})
        for item in data.get('segmentations', []) or []:
            channel = self.one_based_to_zero_based(item.get('channel'))
            if channel is not None:
                plan['segmentations'].append({
                    'object': self.safe_metric_name(item.get('object', 'structure')),
                    'channel': channel,
                    'method': item.get('method', 'llm_custom') or 'llm_custom'
                })
        for item in data.get('cell_segmentations', []) or []:
            channel = self.one_based_to_zero_based(item.get('channel'))
            if channel is not None:
                plan.setdefault('cell_segmentations', []).append({'object': 'cell', 'channel': channel, 'method': 'cellpose'})
        for item in data.get('colocalizations', []) or []:
            source = self.one_based_to_zero_based(item.get('source_channel'))
            target = self.one_based_to_zero_based(item.get('target_channel'))
            if source is not None and target is not None:
                plan['colocalizations'].append({'source_channel': source, 'target_channel': target,
                                                'metric': item.get('metric', 'ratio') or 'ratio'})
        nuclear_channel = self.one_based_to_zero_based(data.get('nuclear_channel'))
        if nuclear_channel is not None:
            plan['nuclear_channel'] = nuclear_channel
        return plan

    def one_based_to_zero_based(self, value):
        if value is None or value == '':
            return None
        try:
            channel = int(value) - 1
        except Exception:
            return None
        if 0 <= channel <= 3:
            return channel
        return None





    def prepare_multichannel_image(self, img):
        img = np.squeeze(img)
        if len(img.shape) > 2 and img.shape[0] > 10:
            img = np.transpose(img, (2, 0, 1))
        return img

    def ensure_cell_mask(self, mask, img_for_channels, img_path, center_cells=None, segmentation_record=None):
        if mask is None:
            seg_channel = self.get_segmentation_channel(segmentation_record, img_for_channels)
            try:
                mask = self.cell_segmentation_without_gui(img_for_channels[seg_channel, :, :], img_path,
                                                          segmentation_record=segmentation_record)
                center_cells = self.get_center_cells(mask)
                if len(center_cells) == 0:
                    raise ValueError('cell segmentation returned no center cells')
                return mask, center_cells
            except Exception as e:
                warnings.warn(f'Cell segmentation failed in analysis; using whole image as one cell: {e}')
                mask = np.ones(img_for_channels.shape[1:], dtype=np.uint8)
                return mask, [1]
        if center_cells is None:
            center_cells = self.get_center_cells(mask)
            if len(center_cells) == 0:
                center_cells = [1] if np.max(mask) >= 1 else []
        return mask, center_cells

    def get_segmentation_channel(self, segmentation_record, img_for_channels):
        if segmentation_record is not None:
            seg_channel = segmentation_record.get('cell_channel', None)
            if seg_channel is not None and seg_channel >= 0:
                return min(seg_channel, img_for_channels.shape[0] - 1)
        seg_channel = self.chan0 if hasattr(self, 'chan0') and self.chan0 is not None and self.chan0 >= 0 else 0
        return min(seg_channel, img_for_channels.shape[0] - 1)

    def cell_segmentation_without_gui(self, img, file=None, segmentation_record=None, channel=None):
        segmentation_record = segmentation_record or {}

        resize_scale = segmentation_record.get('resize_scale', self.resize_scale)
        cell_min_size = segmentation_record.get('cell_min_size', self.cell_min_size)
        model_path = segmentation_record.get('model_path') or cellpose_models.default_cellpose_model_path()
        model_type = segmentation_record.get('model_type', None)
        model_key = (model_path, model_type, segmentation_record.get('use_gpu', cellpose_core.use_gpu()))
        if self.default_cellpose_model is None or self.default_cellpose_model_key != model_key:
            if model_path:
                self.default_cellpose_model = cellpose_models.CellposeModel(
                    gpu=segmentation_record.get('use_gpu', cellpose_core.use_gpu()), pretrained_model=model_path)
            else:
                self.default_cellpose_model = cellpose_models.CellposeModel(
                    gpu=segmentation_record.get('use_gpu', cellpose_core.use_gpu()), model_type=model_type or 'cyto3')
            self.default_cellpose_model_key = model_key
        image = cv2.resize(img, None, fx=1. / resize_scale, fy=1. / resize_scale,
                           interpolation=cv2.INTER_LINEAR)
        min_size = cell_min_size // (resize_scale ** 2)
        niter = segmentation_record.get('niter', 0)
        niter = None if niter == 0 else niter
        masks, flows = self.default_cellpose_model.eval(
            image, channels=segmentation_record.get('channels', [0, 0]),
            diameter=segmentation_record.get('diameter', 30),
            cellprob_threshold=segmentation_record.get('cellprob_threshold', 0.0),
            min_size=min_size,
            flow_threshold=segmentation_record.get('flow_threshold', 0.4),
            do_3D=segmentation_record.get('do_3D', False),
            niter=niter,
            normalize=segmentation_record.get('normalize_params', True),
            stitch_threshold=segmentation_record.get('stitch_threshold', 0.0),
        )[:2]
        masks = masks.astype(np.uint8)
        masks = cv2.resize(masks, None, fx=resize_scale, fy=resize_scale,
                           interpolation=cv2.INTER_NEAREST)
        if file is not None:
            self.save_analysis_cell_mask(masks, img, file, channel=channel)

        return masks

    def save_analysis_cell_mask(self, masks, img, file, channel=None):
        colored_mask = label2rgb(masks, change_to_8bit(img.copy()), bg_label=0) * 255
        channel_suffix = '' if channel is None else f'_chan{channel + 1}'
        overlay_path = os.path.join(
            os.path.dirname(file),
            f'{_image_stem(file)}_analysis_cell_seg{channel_suffix}.png')
        overlay_bgr = cv2.cvtColor(colored_mask.astype(np.uint8), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(overlay_path, overlay_bgr):
            warnings.warn(f'Could not write cell-segmentation overlay: {overlay_path}')
        np.save(os.path.join(os.path.dirname(file),
                             f'{_image_stem(file)}_analysis_cell_mask{channel_suffix}.npy'),
                masks)



    def execute_analysis_plan(self, img, mask, img_path, center_cells, plan, threshold, distance_threshold, useDL,
                              custom_analysis_mode='LLM Python script', segmentation_record=None):


        img = self.prepare_multichannel_image(img)
        self.organelle_deteciton_results = {}
        self.co_localization_result = {}
        self.script_segmentation_results = {}

        if len(img.shape) == 2:
            img_for_channels = img[np.newaxis, :, :]
        elif len(img.shape) == 4:
            warnings.warn("Currently don not support 4D data")
            return
        else:
            img_for_channels = img

        n_channels = img_for_channels.shape[0]
        overlay_channel = self.get_segmentation_channel(segmentation_record, img_for_channels)
        if type(mask) == str:
            record = np.load(mask, allow_pickle=True).item()
            prev_mask = record.get('mask')
            mask = self.cell_segmentation_track(record, prev_mask, img_for_channels, img_path)
            if self.time_point > 0:
                center_cells = record.get('center_cells')
            else:
                center_cells = self.get_center_cells(mask)
            record['mask'] = mask
            record['center_cells'] = center_cells
            np.save(os.path.join(os.path.dirname(img_path),
                                 os.path.basename(img_path).replace('.ome.tif', '_seg_record.npy')), record)
        else:
            needs_cell_mask = bool(plan.get('detections') or plan.get('segmentations') or plan.get('colocalizations'))
            if needs_cell_mask:
                mask, center_cells = self.ensure_cell_mask(mask, img_for_channels, img_path, center_cells,
                                                           segmentation_record=segmentation_record)


        for cell_segmentation in plan.get('cell_segmentations', []):

            chan = cell_segmentation.get('channel', -1)
            if 0 <= chan < n_channels:
                try:
                    overlay_channel = chan
                    cell_record = dict(segmentation_record or {})
                    cell_record['cell_channel'] = chan
                    mask = self.cell_segmentation_without_gui(img_for_channels[chan, :, :], img_path,
                                                              segmentation_record=cell_record, channel=chan)
                    center_cells = self.get_center_cells(mask)
                    if len(center_cells) == 0:
                        center_cells = [cell for cell in range(1, int(np.max(mask)) + 1)]
                    if len(center_cells) == 0:
                        warnings.warn(f'Cell instance segmentation returned no cells for channel {chan + 1}; using whole image as one cell')
                        mask = np.ones(img_for_channels.shape[1:], dtype=np.uint8)
                        center_cells = [1]
                        self.save_analysis_cell_mask(mask, img_for_channels[chan, :, :], img_path, channel=chan)

                except Exception as e:
                    warnings.warn(f'Cell instance segmentation failed for channel {chan + 1}: {e}')
            else:
                warnings.warn(f"WARNING: specified cell segmentation channel {chan + 1} > channel number")

        if mask is not None:
            # Analysis commonly receives a mask generated by the GUI. Save that mask too,
            # so every Run leaves a visual segmentation record beside the source image.
            self.save_analysis_cell_mask(mask, img_for_channels[overlay_channel, :, :], img_path)

        nuclear_channel = plan.get('nuclear_channel', -1)

        nuclear_image = img_for_channels[nuclear_channel, :, :] if 0 <= nuclear_channel < n_channels else None

        for segmentation in plan.get('segmentations', []):

            chan = segmentation.get('channel', -1)
            if 0 <= chan < n_channels:
                result = self.custom_structure_analysis(img_for_channels[chan, :, :], mask, img_path,
                                                        channel=chan,
                                                        object_name=segmentation.get('object', 'structure'),
                                                        center_cells=center_cells,
                                                        instruction=plan.get('instruction', ''),
                                                        custom_analysis_mode=custom_analysis_mode)
                self.script_segmentation_results[(chan, segmentation.get('object', 'structure'))] = result
            else:
                warnings.warn(f"WARNING: specified segmentation channel {chan + 1} > channel number")


        for detection in plan.get('detections', []):
            chan = detection.get('channel', -1)
            if 0 <= chan < n_channels:
                blobs, nuclear_distances, intensities, areas = self.puncta_detection(
                    img_for_channels[chan, :, :], mask, img_path,
                    channel=chan, nuclear_image=nuclear_image, threshold=threshold,
                    center_cells=center_cells, useDL=useDL)
                self.write_basic_property(blobs, img_path, chan, nuclear_distances,
                                          img_for_channels[chan, :, :], intensities, areas)
                self.organelle_deteciton_results[chan] = (blobs, nuclear_distances, intensities)
            else:
                warnings.warn(f"WARNING: specified lysosome channel {chan + 1} > channel number")

        for colocalization in plan.get('colocalizations', []):
            first_chan = colocalization.get('source_channel', -1)
            second_chan = colocalization.get('target_channel', -1)
            if first_chan in self.organelle_deteciton_results and second_chan in self.organelle_deteciton_results:
                co_intensities, co_records = self.calculate_distance(
                    self.organelle_deteciton_results[second_chan][0],
                    self.organelle_deteciton_results[first_chan][0],
                    self.organelle_deteciton_results[first_chan][2],
                    distance_threshold=distance_threshold)
                self.co_localization_result[f'chan{first_chan} in chan{second_chan}'] = (co_intensities, co_records)
                self.write_co_localization(first_chan, second_chan, img_path,
                                           self.organelle_deteciton_results[first_chan][0],
                                           self.organelle_deteciton_results[second_chan][0],
                                           co_intensities, co_records,
                                           distance_threshold=distance_threshold)
            else:
                self.write_mask_colocalization(first_chan, second_chan, img_path)

    def custom_structure_analysis(self, img, masks, file, channel, object_name, center_cells=None,
                                  instruction='', custom_analysis_mode='LLM Python script'):
        object_name = self.safe_metric_name(object_name)
        analysis_dir = self.custom_analysis_dir(file)
        os.makedirs(analysis_dir, exist_ok=True)
        try:
            if custom_analysis_mode == 'LLM FIJI macro':
                mask_result = self.run_llm_fiji_macro(img, masks, file, channel, object_name,
                                                      instruction, analysis_dir)
            else:
                mask_result = self.run_llm_python_script(img, masks, file, channel, object_name,
                                                         instruction, analysis_dir)
        except Exception as e:
            feedback = self.format_llm_feedback(e, context=f'custom analysis generation for {object_name} channel {channel + 1}')
            warnings.warn(feedback)
            logger.warning(feedback)
            self.write_llm_failure_feedback(file, feedback)
            mask_result = np.zeros_like(img, dtype=np.uint8)
        return self.mask_result_to_cell_results(mask_result, img, masks, file, channel, object_name, center_cells)


    def custom_analysis_dir(self, file):
        norm_file = os.path.normpath(file)
        parts = norm_file.split(os.sep)
        if self.time_point >= 0 and len(parts) >= 2:
            experiment_root = os.sep.join(parts[:-2])
        else:
            experiment_root = os.path.dirname(norm_file)
        if experiment_root == '':
            experiment_root = os.path.dirname(norm_file)
        return os.path.join(experiment_root, 'llm_custom_analysis')

    def custom_analysis_hash(self, instruction, channel, object_name, mode):
        payload = {
            'instruction': instruction,
            'channel': channel + 1,
            'object': object_name,
            'mode': mode,
            'contract_version': 1,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:12]

    def run_llm_python_script(self, img, masks, file, channel, object_name, instruction, analysis_dir):
        analysis_id = self.custom_analysis_hash(instruction, channel, object_name, 'python')
        script_path = os.path.join(analysis_dir, f'custom_analysis_{analysis_id}.py')
        metadata_path = os.path.join(analysis_dir, f'custom_analysis_{analysis_id}.json')
        if not os.path.exists(script_path):
            code = self.generate_python_analysis_script(instruction, channel, object_name)
            self.validate_generated_python(code)
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)
            self.write_custom_analysis_metadata(metadata_path, instruction, channel, object_name, 'LLM Python script', script_path)

        self.validate_generated_python_path(script_path)
        output_dir = os.path.join(os.path.dirname(file), 'llm_custom_analysis_outputs')
        os.makedirs(output_dir, exist_ok=True)
        output_mask_path = os.path.join(
            output_dir, f'{object_name}_chan{channel + 1}_time{self.time_point}_mask.npy')
        context = {
            'instruction': instruction,
            'object_name': object_name,
            'channel': channel,
            'channel_one_based': channel + 1,
            'time_point': self.time_point,
            'image_path': file,
            'output_mask_path': output_mask_path,
        }
        module_name = f'cellquant_custom_analysis_{analysis_id}'
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, 'analyze'):
            raise ValueError(f'Generated custom analysis script must define analyze(): {script_path}')
        result = module.analyze(img.copy(), masks.copy(), output_dir, context)
        mask_result = self.extract_mask_from_custom_result(result)
        if mask_result is None and os.path.exists(output_mask_path):
            mask_result = np.load(output_mask_path, allow_pickle=True)
        if mask_result is None:
            warnings.warn(f'Custom Python analysis did not return or save a mask: {script_path}')
            mask_result = np.zeros_like(img, dtype=np.uint8)
        return mask_result

    def run_llm_fiji_macro(self, img, masks, file, channel, object_name, instruction, analysis_dir):
        analysis_id = self.custom_analysis_hash(instruction, channel, object_name, 'fiji')
        macro_path = os.path.join(analysis_dir, f'custom_analysis_{analysis_id}.ijm')
        metadata_path = os.path.join(analysis_dir, f'custom_analysis_{analysis_id}.json')
        if not os.path.exists(macro_path):
            macro = self.generate_fiji_analysis_macro(instruction, channel, object_name)
            with open(macro_path, 'w', encoding='utf-8') as f:
                f.write(macro)
            self.write_custom_analysis_metadata(metadata_path, instruction, channel, object_name, 'LLM FIJI macro', macro_path)

        output_dir = os.path.join(os.path.dirname(file), 'llm_custom_analysis_outputs')
        os.makedirs(output_dir, exist_ok=True)
        input_image_path = os.path.join(output_dir, f'{object_name}_chan{channel + 1}_time{self.time_point}_input.tif')
        cell_mask_path = os.path.join(output_dir, f'{object_name}_chan{channel + 1}_time{self.time_point}_cell_mask.tif')
        output_mask_path = os.path.join(output_dir, f'{object_name}_chan{channel + 1}_time{self.time_point}_fiji_mask.tif')
        cv2.imwrite(input_image_path, img.astype(np.uint16))
        cv2.imwrite(cell_mask_path, masks.astype(np.uint16))
        fiji_executable = self.get_fiji_executable()
        if fiji_executable is None:
            warnings.warn(
                f'FIJI macro generated at {macro_path}, but FIJI_PATH or IMAGEJ_PATH is not configured; skipping execution.')
            return np.zeros_like(img, dtype=np.uint8)
        macro_args = '|'.join([input_image_path, cell_mask_path, output_mask_path, output_dir,
                               json.dumps({'instruction': instruction, 'object_name': object_name, 'channel': channel + 1})])
        try:
            subprocess.run([fiji_executable, '--headless', '-macro', macro_path, macro_args],
                           check=True, timeout=300)
        except Exception as e:
            warnings.warn(f'FIJI macro execution failed: {e}')
            return np.zeros_like(img, dtype=np.uint8)
        if os.path.exists(output_mask_path):
            mask_result = cv2.imread(output_mask_path, cv2.IMREAD_UNCHANGED)
            if mask_result is not None:
                return mask_result
        warnings.warn(f'FIJI macro did not create output mask: {output_mask_path}')
        return np.zeros_like(img, dtype=np.uint8)

    def generate_python_analysis_script(self, instruction, channel, object_name):
        prompt = f"""
You are writing a reusable Python image-analysis script for Micro-copilot real-time microscopy.
Task instruction: {instruction}
Target object: {object_name}
Target channel: channel {channel + 1} (zero-based index {channel})

Return ONLY valid Python code, no markdown fences and no explanation.
The code MUST define exactly this function:

def analyze(image, cell_mask, output_dir, context):
    ...

Inputs:
- image: 2D numpy array for the target channel.
- cell_mask: 2D numpy labeled cell mask; 0 is background, positive integers are cell labels.
- output_dir: directory where optional outputs may be saved.
- context: dict with instruction, object_name, channel, channel_one_based, time_point, image_path, output_mask_path.

Requirements:
- Use only numpy, cv2, scipy, skimage, pandas, os, math if needed.
- Do not call network, shell, subprocess, eval, exec, delete files, or read unrelated files.
- Produce a 2D binary or labeled mask with the same height and width as image.
- Save the mask to context['output_mask_path'] using numpy.save.
- Return a dict containing at least {{'mask': mask}}.
- The script should be reusable for every frame in this experiment; do not hardcode filenames.
""".strip()
        return self.extract_code_block(self.call_llm(prompt))

    def generate_fiji_analysis_macro(self, instruction, channel, object_name):
        prompt = f"""
You are writing a reusable FIJI/ImageJ macro for Micro-copilot real-time microscopy.
Task instruction: {instruction}
Target object: {object_name}
Target channel: channel {channel + 1}

Return ONLY valid ImageJ macro code, no markdown fences and no explanation.
The macro will run headless with one argument string:
input_image_path|cell_mask_path|output_mask_path|output_dir|context_json

Requirements:
- Parse getArgument() split by '|'.
- Open input_image_path.
- Analyze/segment the target object according to the instruction.
- Save a binary or labeled mask image to output_mask_path.
- Do not delete files or require manual UI interaction.
- The macro should be reusable for every frame in this experiment; do not hardcode filenames.
""".strip()
        return self.extract_code_block(self.call_llm(prompt))

    def call_llm(self, prompt, system_content='You write concise, safe, reusable microscopy image-analysis code.'):
        from cellquant.llm_config import require_llm_config
        config = require_llm_config()

        try:
            from openai import AzureOpenAI
        except Exception as e:
            raise RuntimeError(f'openai package is required in the current environment: {e}')
        try:
            client = AzureOpenAI(api_key=config['api_key'], azure_endpoint=config['endpoint'],
                                 api_version=config['api_version'])
            completion = client.chat.completions.create(
                model=config['model'],
                messages=[
                    {'role': 'system', 'content': system_content},
                    {'role': 'user', 'content': prompt},
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise RuntimeError(self.format_llm_feedback(e, context='LLM API call'))

    def format_llm_feedback(self, error, context='LLM request'):
        try:
            from cellquant.llm_config import format_llm_exception
            return format_llm_exception(error, context=context)
        except Exception:
            return f'{context} failed: {error}'

    def write_llm_failure_feedback(self, file, feedback):
        try:
            feedback_dir = os.path.dirname(file) if file else os.getcwd()
            with open(os.path.join(feedback_dir, 'llm_analysis_feedback.txt'), 'a', encoding='utf-8') as f:
                f.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {feedback}\n')
        except Exception:
            pass

    def extract_code_block(self, text):

        match = re.search(r'```(?:python|ijm|java|javascript|imagej)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    def validate_generated_python_path(self, script_path):
        with open(script_path, 'r', encoding='utf-8') as f:
            self.validate_generated_python(f.read())

    def validate_generated_python(self, code):
        forbidden = [
            'subprocess', 'os.system', 'shutil.rmtree', 'socket', 'requests', 'urllib',
            'eval(', 'exec(', '__import__', 'importlib', 'os.remove', '.unlink(', 'rmtree('
        ]


        for item in forbidden:
            if item in code:
                raise ValueError(f'Generated custom analysis script contains forbidden code: {item}')
        if 'def analyze(' not in code:
            raise ValueError('Generated custom analysis script must define analyze(image, cell_mask, output_dir, context).')

    def write_custom_analysis_metadata(self, metadata_path, instruction, channel, object_name, mode, artifact_path):
        metadata = {
            'instruction': instruction,
            'channel': channel + 1,
            'object_name': object_name,
            'mode': mode,
            'artifact_path': artifact_path,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def get_fiji_executable(self):
        for env_name in ['FIJI_PATH', 'IMAGEJ_PATH']:
            value = os.environ.get(env_name)
            if value and os.path.exists(value):
                return value
        return None

    def extract_mask_from_custom_result(self, result):
        if result is None:
            return None
        if isinstance(result, np.ndarray):
            return result
        if isinstance(result, dict):
            mask = None
            for key in ('mask', 'segmentation', 'labels'):
                if key in result and result[key] is not None:
                    mask = result[key]
                    break
            if isinstance(mask, str):
                if mask.endswith('.npy') and os.path.exists(mask):
                    return np.load(mask, allow_pickle=True)
                if os.path.exists(mask):
                    return cv2.imread(mask, cv2.IMREAD_UNCHANGED)
            return mask

        return None

    def mask_result_to_cell_results(self, mask_result, img, masks, file, channel, object_name, center_cells=None):
        object_name = self.safe_metric_name(object_name)
        if center_cells is None:
            center_cells = self.get_center_cells(masks)
        if mask_result is None:
            mask_result = np.zeros_like(img, dtype=np.uint8)
        mask_result = np.squeeze(mask_result)
        if mask_result.shape != img.shape:
            warnings.warn(f'Custom analysis mask shape {mask_result.shape} does not match image shape {img.shape}')
            mask_result = np.zeros_like(img, dtype=np.uint8)
        object_mask_all = mask_result > 0
        segmentation_image = np.zeros_like(img, dtype=np.uint16)
        cell_results = []
        for cell_index, num_cell in enumerate(center_cells, start=1):
            current_cell_mask = masks == num_cell
            object_mask = object_mask_all & current_cell_mask
            labels, number = scipy.ndimage.label(object_mask)
            if np.any(labels > 0):
                segmentation_image[labels > 0] = labels[labels > 0] + np.max(segmentation_image)
            areas = []
            intensities = []
            for obj_id in range(1, number + 1):
                obj_mask = labels == obj_id
                if np.sum(obj_mask) == 0:
                    continue
                areas.append(int(np.sum(obj_mask)))
                intensities.append(float(np.mean(img[obj_mask])))
            cell_results.append({'mask': object_mask, 'areas': areas, 'intensities': intensities})
            self.write_intensity_segmentation_property(file, channel, object_name, cell_index, areas, intensities)

        draw_image = label2rgb(segmentation_image, change_to_8bit(img.copy()), bg_label=0) * 255
        cv2.imwrite(get_unique_filename(os.path.dirname(file),
                                        f'{_image_stem(file)}_{object_name}_chan{channel + 1}_llm_seg.png'),
                    draw_image.astype(np.uint8))
        return {'object': object_name, 'channel': channel, 'cells': cell_results}

    def intensity_segmentation(self, img, masks, file, channel, object_name, center_cells=None):

        object_name = self.safe_metric_name(object_name)
        if center_cells is None:
            center_cells = self.get_center_cells(masks)
        img_for_seg = filters.gaussian(remove_background(img.copy()), sigma=1)
        segmentation_image = np.zeros_like(img_for_seg, dtype=np.uint16)
        cell_results = []
        for cell_index, num_cell in enumerate(center_cells, start=1):
            current_cell_mask = (masks == num_cell)
            if np.sum(current_cell_mask) == 0:
                cell_results.append({'mask': np.zeros_like(current_cell_mask), 'areas': [], 'intensities': []})
                continue
            cell_image = normalize_mask(img_for_seg.copy(), current_cell_mask)
            valid_pixels = cell_image[current_cell_mask]
            if valid_pixels.size == 0 or np.max(valid_pixels) <= np.min(valid_pixels):
                object_mask = np.zeros_like(current_cell_mask)
            else:
                thresh = threshold_otsu(valid_pixels)
                object_mask = (cell_image > thresh) & current_cell_mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                object_mask = cv2.morphologyEx(object_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
            labels, number = scipy.ndimage.label(object_mask)
            segmentation_image[labels > 0] = labels[labels > 0] + np.max(segmentation_image)
            areas = []
            intensities = []
            for obj_id in range(1, number + 1):
                obj_mask = labels == obj_id
                if np.sum(obj_mask) == 0:
                    continue
                areas.append(int(np.sum(obj_mask)))
                intensities.append(float(np.mean(img[obj_mask])))
            cell_results.append({'mask': object_mask, 'areas': areas, 'intensities': intensities})
            self.write_intensity_segmentation_property(file, channel, object_name, cell_index, areas, intensities)

        draw_image = label2rgb(segmentation_image, change_to_8bit(img.copy()), bg_label=0) * 255
        cv2.imwrite(get_unique_filename(os.path.dirname(file),
                                        f'{_image_stem(file)}_{object_name}_chan{channel + 1}_seg.png'),
                    draw_image.astype(np.uint8))
        return {'object': object_name, 'channel': channel, 'cells': cell_results}

    def write_intensity_segmentation_property(self, file, channel, object_name, cell_count, areas, intensities):
        csv_file_name = os.path.join(os.path.dirname(file),
                                     f'{_image_stem(file)}_{object_name}_chan{channel + 1}_cell{cell_count}_time{self.time_point}.csv')
        df = pd.DataFrame({'pixel numbers': areas, 'intensity': intensities})
        try:
            df.to_csv(csv_file_name, index=False)
        except:
            warnings.warn(f'WARN: writing {csv_file_name} csv failed')
        if self.time_point >= 0:
            statistics_csv_file = os.path.join(os.sep.join(os.path.normpath(file).split(os.sep)[:-2]),
                                               f'statistic_time{self.time_point}.csv')
            if not os.path.exists(statistics_csv_file):
                stat_df = pd.DataFrame()
                stat_df['Name'] = None
            else:
                stat_df = pd.read_csv(statistics_csv_file)
            name = os.path.normpath(file).split(os.sep)[-2] + f'_cell{cell_count}'
            if name in stat_df['Name'].values:
                row_index = (stat_df['Name'] == name)
            else:
                row_index = len(stat_df)
                stat_df.loc[row_index, 'Name'] = name
            stat_df.loc[row_index, f'chan{channel + 1} {object_name} number'] = len(areas)
            if areas:
                stat_df.loc[row_index, f'chan{channel + 1} {object_name} area'] = np.sum(np.array(areas))
                stat_df.loc[row_index, f'chan{channel + 1} {object_name} intensity'] = np.mean(np.array(intensities))
            try:
                stat_df.to_csv(statistics_csv_file, index=False)
            except:
                warnings.warn(f'WARN: writing {statistics_csv_file} csv failed')

    def write_mask_colocalization(self, source_channel, target_channel, file):
        source_result = self._first_script_result_for_channel(source_channel)
        target_result = self._first_script_result_for_channel(target_channel)
        if source_result is None or target_result is None:
            return
        if self.time_point < 0:
            return
        statistics_csv_file = os.path.join(os.sep.join(os.path.normpath(file).split(os.sep)[:-2]),
                                           f'statistic_time{self.time_point}.csv')
        if not os.path.exists(statistics_csv_file):
            stat_df = pd.DataFrame()
            stat_df['Name'] = None
        else:
            stat_df = pd.read_csv(statistics_csv_file)
        for cell_count, (source_cell, target_cell) in enumerate(zip(source_result['cells'], target_result['cells']), start=1):
            source_mask = source_cell['mask']
            target_mask = target_cell['mask']
            source_area = np.sum(source_mask)
            ratio = float(np.sum(source_mask & target_mask) / (source_area + 1e-7))
            name = os.path.normpath(file).split(os.sep)[-2] + f'_cell{cell_count}'
            if name in stat_df['Name'].values:
                row_index = (stat_df['Name'] == name)
            else:
                row_index = len(stat_df)
                stat_df.loc[row_index, 'Name'] = name
            stat_df.loc[row_index, f'chan{source_channel + 1} in chan{target_channel + 1} mask co-localization ratio'] = ratio
        try:
            stat_df.to_csv(statistics_csv_file, index=False)
        except:
            warnings.warn(f'WARN: writing {statistics_csv_file} csv failed')

    def _first_script_result_for_channel(self, channel):
        for (result_channel, _), result in self.script_segmentation_results.items():
            if result_channel == channel:
                return result
        return None

    def safe_metric_name(self, name):
        name = re.sub(r'[^0-9a-zA-Z_]+', '_', str(name).strip().lower())
        return name.strip('_') or 'structure'

    def run(self):
        classifier_path = cellpose_models.default_puncta_model_path()
        self.puncta_classifier = (
            PunctaClassifier(model_path=classifier_path)
            if classifier_path and os.path.isfile(classifier_path)
            else None
        )
        while True:
            payload = self.queue.get()
            if payload is None:
                break
            analysis_instruction = ''

            custom_analysis_mode = 'LLM Python script'
            segmentation_record = None
            if not isinstance(payload, (tuple, list)):
                warnings.warn(f'WARNING: invalid analysis queue payload type {type(payload)}')
                continue
            if len(payload) == 11:


                img, mask, img_path, center_cells, (
                    self.chan0, self.chan1, self.chan2,
                    nuclear_chan), threshold, distance_threshold, useDL, time_point, draw_x, draw_y = payload
            elif len(payload) == 12:
                img, mask, img_path, center_cells, (
                    self.chan0, self.chan1, self.chan2,
                    nuclear_chan), threshold, distance_threshold, useDL, time_point, draw_x, draw_y, analysis_instruction = payload
            elif len(payload) == 13:
                img, mask, img_path, center_cells, (
                    self.chan0, self.chan1, self.chan2,
                    nuclear_chan), threshold, distance_threshold, useDL, time_point, draw_x, draw_y, analysis_instruction, custom_analysis_mode = payload
            elif len(payload) >= 14:
                img, mask, img_path, center_cells, (
                    self.chan0, self.chan1, self.chan2,
                    nuclear_chan), threshold, distance_threshold, useDL, time_point, draw_x, draw_y, analysis_instruction, custom_analysis_mode, segmentation_record = payload[:14]
            else:

                warnings.warn('WARNING: invalid analysis queue payload')
                continue


            logger.info(f"analysis queue get: {img_path}")
            print(f"analysis queue get: {img_path}")
            self.time_point = time_point
            if img_path is None and img is None:
                break

            if self.chan1 == self.chan0:
                self.chan1 = -1
            if self.chan2 == self.chan0:
                self.chan2 = -1
            if self.chan2 == self.chan1:
                self.chan2 = -1

            detection_channels = [self.chan0, self.chan1, self.chan2]
            plan = self.get_analysis_plan(analysis_instruction, detection_channels, nuclear_chan)
            logger.info(f"analysis plan: {plan}")
            try:
                self.execute_analysis_plan(img, mask, img_path, center_cells, plan,
                                           threshold, distance_threshold, useDL, custom_analysis_mode,
                                           segmentation_record=segmentation_record)


                if self.should_send_to_plot_queue(draw_x, draw_y):
                    self.plot(draw_x, draw_y)
                self.result_queue.put((self.organelle_deteciton_results, self.co_localization_result))

            except Exception as e:
                warnings.warn(f'Analysis failed: {e}')
            finally:
                self.done_event.set()



    def should_send_to_plot_queue(self, drawx, drawy):
        drawx = (drawx or '').strip(' ')
        drawy = (drawy or '').strip(' ')
        if 'none' in drawx and 'none' in drawy:
            return False
        if not self.organelle_deteciton_results:
            return False
        return self.is_cellquant_plot_order(drawx) and self.is_cellquant_plot_order(drawy)

    def is_cellquant_plot_order(self, draw_order):
        draw_order = (draw_order or '').strip(' ')
        if draw_order == '' or draw_order == 'none' or draw_order == 'time':
            return True
        try:
            first_chan = int(draw_order[1]) - 1
        except Exception:
            return False
        if first_chan not in self.organelle_deteciton_results:
            return False
        if 'in' in draw_order:
            try:
                second_chan = int(draw_order.split(' ')[2][1]) - 1
            except Exception:
                return False
            return f'chan{first_chan} in chan{second_chan}' in self.co_localization_result
        return True

    def plot(self, drawx, drawy):
        drawx = drawx.strip(' ')

        drawy = drawy.strip(' ')
        logger.info(f"want to plot {drawx}, {drawy}")
        if 'none' in drawx and 'none' in drawy:
            return
        out_x = self.draw(drawx)
        out_y = self.draw(drawy)
        if len(out_x) != len(out_y):
            if drawx == 'time':
                out_x = [self.time_point for _ in range(len(out_y))]
            elif drawx == 'none':
                out_x = [0 for _ in range(len(out_y))]
            elif drawy == 'none':
                out_y = [0 for _ in range(len(out_x))]
            else:
                warnings.warn('Fail to draw the cell level information and structure level information '
                              'or different channel information on on the same figure')
                out_x = []
                out_y = []
        logger.info(f"'out_x:', {len(out_x)}, 'outy:', {len(out_y)}")
        if self.time_point >= 0:
            self.plot_queue.put((out_x, out_y, False, drawx, drawy))
        else:
            self.plot_queue.put((out_x, out_y, True, drawx, drawy))
        # draw y

    def draw(self, draw_order):  # make it more entensible
        out = []
        if draw_order == 'none':
            return out
        if draw_order == 'time':
            out.append(self.time_point)
            return out

        first_chan = int(draw_order[1]) - 1
        blobs, nuclear_distances, intensities = self.organelle_deteciton_results.get(first_chan, (None, None, None))
        if blobs is None and nuclear_distances is None:
            warnings.warn('Please choose the channels that are used for detection')
            return out
        if 'co-localization ratio' in draw_order:  # ratio, cell level draw
            assert 'in' in draw_order
            second_chan = int(draw_order.split(' ')[2][1]) - 1
            co_intensities, co_records = self.co_localization_result[f'chan{first_chan} in chan{second_chan}']
            for blob, co_record in zip(blobs, co_records):
                out.append(np.array(np.sum(np.array(co_record))) / (np.array(len(blob)) + 1e-7))
        elif 'structure intensity' in draw_order:  # intensity, organelle wise
            chan = draw_order.split('structure')[0]
            if 'in' in chan:  # co
                second_chan = int(draw_order.split(' ')[2][1]) - 1
                co_intensities, co_records = self.co_localization_result[f'chan{first_chan} in chan{second_chan}']
                for co_intensity in co_intensities:
                    out.extend(co_intensity)
            else:  # not co
                for intensity in intensities:
                    out.extend(intensity)
        elif 'structure number' in draw_order:  # number, cell wise
            chan = draw_order.split('structure')[0]
            if 'in' in chan:  # co
                second_chan = int(draw_order.split(' ')[2][1]) - 1
                co_intensities, co_records = self.co_localization_result[f'chan{first_chan} in chan{second_chan}']
                for co_record in co_records:
                    out.append(np.sum(np.array(co_record)))
            else:  # not co
                for blob in blobs:
                    out.append(len(blob))
        elif 'structure radius' in draw_order:  # organelle wise
            for blob in blobs:
                for i in range(len(blob)):
                    out.append(blob[i][2] * 1.4)
        elif 'structure nuclear distance' in draw_order:  # organelle wise
            for nuclear_distance in nuclear_distances:
                if None in nuclear_distance:
                    warnings.warn('Not calculate nuclear distance, but want to draw')
                    return []
                for i in range(len(nuclear_distance)):
                    out.append(nuclear_distance[i])
        return out

    def write_basic_property(self, blobs, file, channel, nuclear_distances, img, all_intensities, all_areas):
        # [] for no cell, [[]] for no organelle
        cell_count = 1
        for blob, nuclear_dis, intensities, areas in zip(blobs, nuclear_distances, all_intensities, all_areas):
            csv_file_name = os.path.join(os.path.dirname(file),
                                         f'{_image_stem(file)}_structure_chan{channel + 1}_cell{cell_count}_time{self.time_point}.csv')
            df = pd.DataFrame()
            df[['position', 'pixel numbers', 'nuclear distance', 'intensity']] = None
            for i in range(len(blob)):
                y, x, r = blob[i]
                perinuclear_dis = 'None'
                if nuclear_dis[i] is not None:
                    perinuclear_dis = str(nuclear_dis[i])
                position = f"({y}, {x})"
                record = pd.DataFrame(
                    {'position': position, 'pixel numbers': areas[i], 'nuclear distance': perinuclear_dis,
                     'intensity': [intensities[i]]})
                df = pd.concat([df, record], ignore_index=True)
                # Write the formatted position and radius to the CSV
            try:
                df.to_csv(csv_file_name, index=False)
            except:
                warnings.warn(f'WARN: writing {csv_file_name} csv failed')
                # break
            if self.time_point >= 0:
                statistics_csv_file = os.path.join(os.sep.join(os.path.normpath(file).split(os.sep)[:-2]),
                                                   f'statistic_time{self.time_point}.csv')
                # write to outer statistic file
                if not os.path.exists(statistics_csv_file):
                    # create a new csv
                    df = pd.DataFrame()
                    df['Name'] = None
                else:
                    df = pd.read_csv(statistics_csv_file)
                name = os.path.normpath(file).split(os.sep)[-2] + f'_cell{cell_count}'
                if name in df['Name'].values:
                    row_index = (df['Name'] == name)
                else:
                    row_index = len(df)
                    df.loc[row_index, 'Name'] = name

                if len(blob) > 0:
                    df.loc[row_index, f'chan{channel + 1} structure pixel number'] = np.mean(
                        np.array(areas))
                    df.loc[row_index, f'chan{channel + 1} structure intensity'] = np.mean(
                        np.array(intensities))
                df.loc[row_index, f'chan{channel + 1} structure number'] = len(blob)
                if None not in nuclear_dis and len(blob) > 0:
                    df.loc[row_index, f'chan{channel + 1} structure nuclear distance'] = np.mean(
                        np.array(nuclear_dis))
                try:
                    df.to_csv(statistics_csv_file, index=False)
                except:
                    warnings.warn(f'WARN: writing {statistics_csv_file} csv failed')
                    # break
            cell_count += 1

    def puncta_segmentation(self, image, blob, segmentation_image, primary_segmentation):
        y, x, r = blob
        # Define cropping box with enlarged radius
        enlarged_r = int(r * 1.4 * 2)
        top_left_y = int(max(0, y - enlarged_r))
        top_left_x = int(max(0, x - enlarged_r))
        bottom_right_y = int(min(image.shape[0], y + enlarged_r))
        bottom_right_x = int(min(image.shape[1], x + enlarged_r))
        # Crop the region
        crop_img = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        # Apply segmentation to the cropped region
        crop_img = gaussian_filter(crop_img, sigma=2)
        thresh = threshold_otsu(crop_img)
        cleared = closing(crop_img > thresh, np.ones((3, 3), dtype=bool))
        targety = int(min(enlarged_r, y))
        targetx = int(min(enlarged_r, x))
        if cleared[targety, targetx] == 0:
            # non object
            rr, cc = disk((int(y), int(x)), int(r * 1.4), shape=image.shape)
            idx = np.max(segmentation_image)
            segmentation_image[rr, cc] = (idx + 1)
            return rr, cc
        else:
            idx = np.max(segmentation_image)
            primary_idx = primary_segmentation[int(y), int(x)]
            object_rows, object_cols = np.where(cleared != 0)
            tmp = np.zeros_like(primary_segmentation)
            tmp[object_rows + top_left_y, object_cols + top_left_x] = 1
            segmentation_image[np.where((tmp == 1) & (segmentation_image != 0))] = (idx + 1)
            return np.where((tmp == 1) & (primary_segmentation == primary_idx))

    def write_co_localization(self, channel1, channel2, file, blobs1, blobs2, co_intensities, co_records,
                              distance_threshold=None):
        # [] for no cells, [[], []] for no organelle
        summary_rows = []
        cell_count = 1
        for blob1, blob2, co_record, co_intensity in zip(blobs1, blobs2, co_records, co_intensities):
            if len(co_record) != len(blob1):
                raise ValueError(
                    f'Co-localization record length {len(co_record)} does not match '
                    f'channel {channel1 + 1} puncta count {len(blob1)}')
            csv_file_name = os.path.join(os.path.dirname(file),
                                         f'{_image_stem(file)}_structure_chan{channel1 + 1}_cell{cell_count}_time{self.time_point}.csv')
            if os.path.exists(csv_file_name):
                df = pd.read_csv(csv_file_name)
            else:
                # Keep the co-localization export usable even if the basic puncta
                # table could not be written earlier in the run.
                df = pd.DataFrame(index=range(len(blob1)))
            for i in range(len(blob1)):
                df.loc[i, f'in chan{channel2 + 1}'] = co_record[i]
            df.to_csv(csv_file_name, index=False)

            coloc_count = int(np.sum(np.asarray(co_record, dtype=np.int64)))
            source_count = len(blob1)
            summary_rows.append({
                'image': os.path.basename(file),
                'time_point': self.time_point,
                'cell': cell_count,
                'source_channel': channel1 + 1,
                'target_channel': channel2 + 1,
                'source_puncta_count': source_count,
                'target_puncta_count': len(blob2),
                'colocalized_puncta_count': coloc_count,
                'colocalization_ratio': coloc_count / source_count if source_count else 0.0,
                'mean_colocalized_intensity': float(np.mean(co_intensity)) if len(co_intensity) else np.nan,
                'distance_threshold': distance_threshold,
            })
            if self.time_point >= 0:
                df = pd.read_csv(os.path.join(os.sep.join(os.path.normpath(file).split(os.sep)[:-2]),
                                              f'statistic_time{self.time_point}.csv'))
                name = os.path.normpath(file).split(os.sep)[-2] + f'_cell{cell_count}'
                if name in df['Name'].values:
                    df.loc[df[
                               'Name'] == name, f'chan{channel1 + 1} in chan{channel2 + 1} co-localization ratio'] = np.array(
                        np.sum(np.array(co_record))) / (np.array(len(blob1)) + 1e-7)
                    df.loc[df['Name'] == name, f'chan{channel1 + 1} in chan{channel2 + 1} structure number'] = np.array(
                        np.sum(np.array(co_record)))
                    if len(co_intensity) > 0:
                        df.loc[
                            df[
                                'Name'] == name, f'chan{channel1 + 1} in chan{channel2 + 1} structure intensity'] = np.mean(
                            np.array(co_intensity))
                try:
                    df.to_csv(os.path.join(os.sep.join(os.path.normpath(file).split(os.sep)[:-2]),
                                           f'statistic_time{self.time_point}.csv'), index=False)
                except:
                    warnings.warn('write cvs failed, please close the csv')
                    # break
            cell_count += 1

        self.write_colocalization_summary(file, summary_rows, channel1, channel2)

    def write_colocalization_summary(self, file, rows, source_channel, target_channel):
        """Upsert a readable, per-image summary beside the microscopy image."""
        if not rows:
            return
        summary_path = os.path.join(os.path.dirname(file), f'{_image_stem(file)}_colocalization.csv')
        new_rows = pd.DataFrame(rows)
        if os.path.exists(summary_path):
            try:
                existing = pd.read_csv(summary_path)
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                existing = pd.DataFrame()
            required_keys = {'time_point', 'cell', 'source_channel', 'target_channel'}
            if required_keys.issubset(existing.columns):
                replace = (
                    (existing['time_point'] == self.time_point)
                    & (existing['source_channel'] == source_channel + 1)
                    & (existing['target_channel'] == target_channel + 1)
                )
                existing = existing.loc[~replace]
            summary = pd.concat([existing, new_rows], ignore_index=True)
        else:
            summary = new_rows
        summary = summary.sort_values(
            ['time_point', 'cell', 'source_channel', 'target_channel'],
            kind='stable').reset_index(drop=True)
        summary.to_csv(summary_path, index=False)
        # if len(blobs1) == 0 and len(blobs2) == 0:
        #     blobs1 = [[]]
        #     blobs2 = [[]]
        #     co_num12 = [0]
        #     co_num21 = [0]
        #     co_intensitys12 = [0]
        #     co_intensitys21 = [0]

        # df = pd.DataFrame(
        #     {f'chan{channel1}_puncta': [np.array(len(blob1)) for blob1 in blobs1],
        #      f'chan{channel2}_puncta': [np.array(len(blob2)) for blob2 in blobs2],
        #      f'chan{channel1} number co with chan{channel2}': [np.array(co_num) for co_num in co_num21],
        #      f'chan{channel2} number co with chan{channel1}': [np.array(co_num) for co_num in co_num12],
        #      f'co/chan{channel1} ratio': [np.array(co_num) / (np.array(len(blob1)) + 1e-7) for co_num, blob1 in
        #                                   zip(co_num21, blobs1)],
        #      f'co/chan{channel2} ratio': [np.array(co_num) / (np.array(len(blob2)) + 1e-7) for co_num, blob2 in
        #                                   zip(co_num12, blobs2)],
        #      f'chan{channel1} intensity co with chan{channel2}': [
        #          np.mean(np.array(co_intensity21)) if len(co_intensity21) > 0 else 0 for co_intensity21 in
        #          co_intensitys21],
        #      f'chan{channel2} intensity co with chan{channel1}': [
        #          np.mean(np.array(co_intensity12)) if len(co_intensity12) > 0 else 0 for co_intensity12 in
        #          co_intensitys12]
        #      })
        # try:
        #     csv_file_name = get_unique_filename(os.path.dirname(file), f'co_chan{channel1}_chan{channel2}.csv')
        #     df.to_csv(csv_file_name, index=False, mode='w')
        # except:
        #     warnings.warn('WARN: writing csv failed')
        # write to statistic file

    def nuclear_segmentation(self, img, file):
        otsu_threshold, image_result = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        image_result = cv2.morphologyEx(image_result, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(os.path.join(os.path.dirname(file), 'nuclear_segmentation.png'), image_result)
        return image_result

    def calculate_nuclear_dis(self, blobs, nuclear_mask, cell_mask, file):
        print('nuclear mask:', nuclear_mask.max(), nuclear_mask.min())
        nuclear_mask[nuclear_mask!=0] = 1
        inter_nuclear_region = cell_mask - nuclear_mask
        dist_from_nuclear = distance_transform_edt(nuclear_mask == 0) * inter_nuclear_region
        max_dist = np.max(dist_from_nuclear)
        region_bounds = np.linspace(0, max_dist, 6)  # Including both edges, so 6 points
        annular_regions = np.digitize(dist_from_nuclear, region_bounds)
        print('annular region:', annular_regions.max(), annular_regions.min())
        cv2.imwrite(os.path.join(os.path.dirname(file), 'annual_region.png'), annular_regions)
        min_distances = []
        for y, x, r in blobs:
            # Ensure the coordinates are within the array bounds
            y, x = int(y), int(x)
            if 0 <= y < dist_from_nuclear.shape[0] and 0 <= x < dist_from_nuclear.shape[1]:
                distance = annular_regions[y, x]
            else:
                distance = np.inf  # or some large number to indicate out of bounds
            min_distances.append(distance)
        # Print or return the min distances
        return min_distances

    def calculate_distance(self, blob_sets1, blob_sets2, intensities2, distance_threshold=4):
        co_nums = []
        co_intensitys = []
        co_records = []
        for blob_set1, blob_set2, intensities in zip(blob_sets1, blob_sets2, intensities2):
            if len(blob_set2) == 0:
                co_nums.append(0)
                co_intensitys.append([])
                co_records.append([])
                continue
            elif len(blob_set1) == 0:
                co_nums.append(0)
                co_intensitys.append([])
                co_records.append([0 for _ in range(len(blob_set2))])
                continue
            co_num = 0
            coords1, radii1 = zip(*[((y, x), r * 1.4) for y, x, r in blob_set1])
            coords2, radii2 = zip(*[((y, x), r * 1.4) for y, x, r in blob_set2])
            tree = KDTree(coords1)

            co_intensity = []
            co_record = []
            # The maximum radius in set2 to determine the search range in the k-d tree
            max_radius_set1 = max(radii1)

            for i, ((y, x), r2) in enumerate(zip(coords2, radii2)):
                # Find potential candidates
                candidates = tree.query_ball_point((y, x), r2 + max_radius_set1 + max(0., distance_threshold))
                min_surface_distance = float('inf')
                for index in candidates:
                    y2, x2 = coords1[index]
                    r1 = radii1[index]
                    center_distance = np.sqrt((y2 - y) ** 2 + (x2 - x) ** 2)
                    surface_distance = center_distance - (r1 + r2)

                    if surface_distance < min_surface_distance:
                        min_surface_distance = surface_distance
                if min_surface_distance < distance_threshold:
                    co_num += 1
                    # rr, cc = disk((int(y), int(x)), int(r2 * 1.4), shape=img2.shape)
                    # intensity = np.mean(img2[rr, cc])
                    intensity = intensities[i]
                    co_intensity.append(intensity)
                    co_record.append(1)
                else:
                    co_record.append(0)
            co_nums.append(co_num)
            co_intensitys.append(co_intensity)
            co_records.append(co_record)
        return co_intensitys, co_records

    def merge_circles(self, circles, threshold=20, max_threshold=40):
        """
        Merges circles that are within `threshold` pixels of each other.

        :param circles: List of circles (y, x, r)
        :param threshold: Distance threshold for merging
        :return: List of merged circles (y, x, r)
        """
        if not circles:
            return []

        # Calculate the pairwise distance matrix
        coords = np.array([(y, x) for y, x, _ in circles])
        dist_matrix = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))

        # Find circles to merge based on the threshold
        to_merge = dist_matrix < threshold
        to_exclude = dist_matrix > max_threshold

        merged = []
        merged_indices = set()

        for i in range(len(circles)):
            if i in merged_indices:
                continue

            # Indices of circles to be merged with the current one
            merge_with = np.where(to_merge[i])[0]
            all_merge_indices = set(merge_with)

            # Check for indirect connections (diffusion process)
            while True:
                new_indices = set(np.where(to_merge[list(all_merge_indices)].any(axis=0))[0])
                new_indices_exclude = set(np.where(to_exclude[list(all_merge_indices)].any(axis=0))[0])
                new_indices = new_indices - new_indices_exclude - all_merge_indices - merged_indices
                if len(new_indices) == 0:
                    break
                all_merge_indices |= new_indices
            y_avg = np.average([circles[j][0] for j in all_merge_indices])
            x_avg = np.average([circles[j][1] for j in all_merge_indices])
            if len(all_merge_indices) == 1:
                r_max = max([circles[j][2] for j in
                             all_merge_indices])
            else:
                r_max = max([math.sqrt((circles[j][0] - y_avg) ** 2 + (circles[j][1] - x_avg) ** 2) for j in
                             all_merge_indices])
                r_max = min(max_threshold // 2, r_max)
            merged.append((y_avg, x_avg, r_max))
            merged_indices |= all_merge_indices

        return merged
