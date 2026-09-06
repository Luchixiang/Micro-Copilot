import warnings

import numpy as np

from cellquant import core as cellpose_core, models as cellpose_models

from skimage.color import lab2rgb, label2rgb
from tifffile import imwrite, imread

from pathlib import Path

from cellquant.utils import *
import warnings

PYCRO_STATUS = 1
try:
    import pycromanager
except:
    PYCRO_STATUS = 0
import random
from multiprocessing import Process, Queue
import time
from shapely.geometry import box
from cellquant.imaged_area_tracker import ImagedAreaTracker

import os
import socket
from qtpy.QtCore import Signal, QObject
from cellquant.writer_process import WriterProcess


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def safe_int(value, default, name='integer'):
    try:
        return int(float(value))
    except Exception:
        warnings.warn(f'Invalid {name}: {value}; using {default}')
        return default


def safe_float(value, default, name='float'):
    try:
        return float(value)
    except Exception:
        warnings.warn(f'Invalid {name}: {value}; using {default}')
        return default


class SmartMicroscope(QObject):

    snap_img_signal = Signal(np.ndarray)
    snap_mask_signal = Signal(np.ndarray)
    finished_signal = Signal(str)

    def __init__(self, ui):
        super().__init__()
        self.resize_scale = 4

        self.ui = ui
        self.queue = self.ui.queue
        global PYCRO_STATUS
        self.cell_min_size = 200 * 200
        self.current_image_array = []
        self.count = 1
        self.capture_cell_count = 0
        self.cell_position_list = []
        self.snap_count = 0
        self.last_movement = 'upper lower cell'
        if PYCRO_STATUS:
            try:
                with socket.create_connection(('127.0.0.1', 4827), timeout=0.2):
                    pass
                self.core = pycromanager.Core()
                self.studio = pycromanager.Studio()

                afm = self.studio.get_autofocus_manager()
                self.afm_method = afm.get_autofocus_method()
                # self.channels = ["7-GFP-mCherry"]
                # self.channel_exposures_ms = [200]
                print('start position:', self.core.get_x_position(), self.core.get_y_position())
                logger.info(f"'start position:', {self.core.get_x_position()}, {self.core.get_y_position()}")
                self.width = self.core.get_image_width()
                self.height = self.core.get_image_height()
                self.pixel_size = self.core.get_pixel_size_um()
                print('get image width and height', self.width, self.height, self.pixel_size)
                logger.info(f"'get image width and height:', {self.width}, {self.height}, {self.pixel_size}")
            except:
                PYCRO_STATUS = 0

        self.last_option = 'init'
        self.last_xmove = 0
        self.last_ymove = 0
        self.imaged_area_tracker = ImagedAreaTracker()
        self.snapped_area_tracker = ImagedAreaTracker()
        self.total_cell_count = 100
        self.image_overlap_ratio_thresh = 0.6
        self.snap_overlap_ratio_thresh = 1.0
        self.close = False

    def imaging(self):
        if not PYCRO_STATUS:
            print(
                "Pycromanager is not working, please ensure it has been installed using 'pip install pycromanager' or ensure the microscope is connected and micro-manager is working")
            return
        self.close = False
        self.get_imaging_settings()
        self.get_save_path()
        # self.get_snap_settings()
        print('start imaging setting:', self.channel_group, self.channels, self.channel_exposures_ms)
        logger.info(f"'start imaging setting:', {self.channel_group}, {self.channels}, {self.channel_exposures_ms}")
        print(' saving directory setting', self.directory, self.name)
        logger.info(f"'saving directory setting:', {self.directory}, {self.name}")
        if os.path.exists(os.path.join(self.directory, 'imaged_area_tracker.txt')):
            with open(os.path.join(self.directory, 'imaged_area_tracker.txt'), 'r') as f:
                for line in f.readlines():
                    corrds = line.split('\t')
                    new_area = box(corrds[0],
                                   corrds[1],
                                   corrds[2],
                                   corrds[3])
                    self.imaged_area_tracker.add_imaged_area(new_area)
            print('loaded pre tracker')
        if os.path.exists(os.path.join(self.directory, 'snap_tracker.txt')):
            with open(os.path.join(self.directory, 'snap_tracker.txt'), 'r') as f:
                for line in f.readlines():
                    corrds = line.split('\t')
                    new_area = box(corrds[0],
                                   corrds[1],
                                   corrds[2],
                                   corrds[3])
                    self.snapped_area_tracker.add_imaged_area(new_area)
        # print('start snapping setting', self.channel_group, self.snap_channel_name, self.snap_exposure_time)
        self.total_cell_count = safe_int(self.ui.cellNumberEdit.text(), 20, 'cell number')
        chan1 = self.ui.punctaChannelChoose[0].currentIndex()

        chan2 = self.ui.punctaChannelChoose[1].currentIndex() - 1
        chan3 = self.ui.punctaChannelChoose[2].currentIndex() - 1
        nuclear_chan = self.ui.punctaChannelChoose[3].currentIndex() - 1
        thresh = self.ui.punctaThreshText.text()
        distance_threshold = self.ui.punctaDistText.text()
        use_analysis = self.ui.analysis_checkbox.isChecked()
        # self.ui.disable_buttons_removeROIs()
        useDL = self.ui.useDL.isChecked()
        self.count = 1
        self.capture_cell_count = 0
        self.snap_count = 0
        draw_x = self.ui.figure_x_combobox_chan.currentText() + ' ' + self.ui.figure_x_combobox_property.currentText()
        draw_y = self.ui.figure_y_combobox_chan.currentText() + ' ' + self.ui.figure_y_combobox_property.currentText()
        analysis_instruction = self.ui.get_analysis_instruction() if hasattr(self.ui, 'get_analysis_instruction') else ''
        custom_analysis_mode = self.ui.get_custom_analysis_mode() if hasattr(self.ui, 'get_custom_analysis_mode') else 'LLM Python script'
        segmentation_record = self.get_segmentation_record(mask=None)
        while self.capture_cell_count < self.total_cell_count and not self.close:



            start_time = time.time()
            # center_cells, upper_cells, right_cells, left_cells, lower_cells, mask, colored_mask = self.snap_and_analysis(
            #     self.directory)
            center_cells, uncenter_cells, mask, colored_mask, image = self.snap_and_analysis(
                self.directory)
            logger.info(f"snapping and segmentation time, {time.time() - start_time}")
            if mask is not None and image is not None:
                self.snap_count += 1
                if len(center_cells) > 0:
                    logger.info(f"{self.snap_count}, have center cells, start imaging")
                    # acquired_dataset = self.acquire_xy_channel()
                    # img = np.stack(self.current_image_array, axis=0)
                    currentx = self.core.get_x_position()
                    currenty = self.core.get_y_position()
                    new_area = box(currentx - self.width * self.pixel_size // 2,
                                   currenty - self.height * self.pixel_size // 2,
                                   currentx + self.width * self.pixel_size // 2,
                                   currenty + self.height * self.pixel_size // 2)
                    self.imaged_area_tracker.add_imaged_area(new_area)
                    with open(os.path.join(self.directory, 'imaged_area_tracker.txt'), 'a') as f:
                        f.write(
                            f'{currentx - self.width * self.pixel_size // 2}\t{currenty - self.height * self.pixel_size // 2} \t'
                            f'{currentx + self.width * self.pixel_size // 2}\t{currenty + self.height * self.pixel_size // 2}')
                        f.write('\n')
                    img_path = os.path.join(self.directory, self.name + '_' + str(self.count),
                                            self.name + '_NDTiffStack.ome.tif')

                    if self.usez:
                        dataset = self.acquire_xyz_channel()
                        image = np.array(dataset.as_array())
                        img = np.max(image, axis=1)
                    else:
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        img = image.copy()
                        imwrite(img_path, img)
                    self.snap_img_signal.emit(img)
                    cv2.imwrite(
                        os.path.join(os.path.dirname(img_path),
                                     os.path.basename(img_path).replace('.tif', '_seg.png')),
                        colored_mask.astype(np.uint8))
                    if use_analysis:
                        self.queue.put(
                            (img, mask, img_path, None, (chan1, chan2, chan3, nuclear_chan),
                             safe_float(thresh, 0.2, 'puncta threshold'),
                             safe_int(distance_threshold, -2, 'co-distance threshold'), useDL, 0, draw_x, draw_y, analysis_instruction,

                             custom_analysis_mode, segmentation_record))



                    self.count += 1
                    self.capture_cell_count += len(center_cells)
            y_distance, x_distance = self.decide_how_to_move(uncenter_cells)
            self.move_microscope([x_distance, y_distance])
        self.finished_signal.emit('finished')
        self.close = False

    def time_lapse_imaging(self):
        self.get_imaging_settings()
        self.get_save_path()
        self.cell_position_list.clear()
        logger.info(f"'start imaging setting:', {self.channel_group}, {self.channels}, {self.channel_exposures_ms}")
        print(' saving directory setting', self.directory, self.name)
        logger.info(f"'saving directory setting:', {self.directory}, {self.name}")
        if os.path.exists(os.path.join(self.directory, 'cell_position_list.txt')):
            with open(os.path.join(self.directory, 'cell_position_list.txt')) as f:
                for line in f.readlines():
                    cur_x, cur_y = line.split('\t')
                    self.cell_position_list.append((float(cur_x), float(cur_y)))
        else:
            self.time_lapse_imaging_stage1()
        # self.move_microscope_absolute(self.cell_position_list[0])
        if len(self.cell_position_list) > 0 and not self.close:
            # constantly face the problem that cannot focus in the first position, so pre-move to that position and focus
            # self.core.full_focus()
            time.sleep(0.5)
            self.afm_method.full_focus()
        self.time_lapse_imaging_stage2()
        # self.ui.writer_queue.put((None,  None, None, None))
        self.close = False

    def time_lapse_imaging_stage1(self):
        if not PYCRO_STATUS:
            print(
                "Pycromanager is not working, please ensure it has been installed using 'pip install pycromanager' or ensure the microscope is connected and micro-manager is working")
            return
        print('start time lapse imaging first stage: find cells')
        logger.info(f"start time lapse imaging first stage: find cells")

        self.total_cell_count = safe_int(self.ui.cellNumberEdit.text(), 20, 'cell number')
        self.count = 1
        self.capture_cell_count = 0
        self.snap_count = 0
        while self.capture_cell_count < self.total_cell_count and not self.close:

            start_time = time.time()
            # center_cells, upper_cells, right_cells, left_cells, lower_cells, mask, colored_mask = self.snap_and_analysis(
            #     self.directory)
            center_cells, uncenter_cells, mask, colored_mask, image = self.snap_and_analysis(self.directory)
            logger.info(f"snapping and segmentation time, {time.time() - start_time}")
            if mask is not None and image is not None:
                self.snap_count += 1
                if len(center_cells) > 0:
                    logger.info(f'captured {self.capture_cell_count}, current snap count {self.snap_count}')
                    currentx = self.core.get_x_position()
                    currenty = self.core.get_y_position()
                    self.cell_position_list.append((currentx, currenty))
                    new_area = box(currentx - self.width * self.pixel_size // 2,
                                   currenty - self.height * self.pixel_size // 2,
                                   currentx + self.width * self.pixel_size // 2,
                                   currenty + self.height * self.pixel_size // 2)
                    self.imaged_area_tracker.add_imaged_area(new_area)
                    with open(os.path.join(self.directory, 'imaged_area_tracker.txt'), 'a') as f:
                        f.write(
                            f'{currentx - self.width * self.pixel_size // 2}\t{currenty - self.height * self.pixel_size // 2} \t'
                            f'{currentx + self.width * self.pixel_size // 2}\t{currenty + self.height * self.pixel_size // 2}')
                        f.write('\n')
                    img_path = os.path.join(self.directory, self.name + '_' + str(self.count),
                                            self.name + '_NDTiffStack.ome.tif')
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    cv2.imwrite(
                        os.path.join(os.path.dirname(img_path),
                                     os.path.basename(img_path).replace('.ome.tif', '_seg.png')),
                        colored_mask.astype(np.uint8))
                    self.save_record(mask, img_path)
                    self.count += 1
                    self.capture_cell_count += len(center_cells)
            y_distance, x_distance = self.decide_how_to_move(uncenter_cells)
            self.move_microscope([x_distance, y_distance])
        with open(os.path.join(self.directory, 'cell_position_list.txt'), 'w') as f:
            for (cur_x, cur_y) in self.cell_position_list:
                f.write(str(cur_x) + '\t' + str(cur_y))
                f.write('\n')

        if len(self.cell_position_list) > 0:
            self.move_microscope_absolute(self.cell_position_list[0])
        # self.finished_signal.emit('finished')
    #
    def time_lapse_imaging_stage2(self):
        if not PYCRO_STATUS:
            print(
                "Pycromanager is not working, please ensure it has been installed using 'pip install pycromanager' or ensure the microscope is connected and micro-manager is working")
            return
        print('start imaging')
        self.get_imaging_settings()
        self.get_save_path()
        self.close = False

        print('start imaging setting:', self.channel_group, self.channels, self.channel_exposures_ms)
        print('start saving directory setting', self.directory, self.name)

        self.total_cell_count = safe_int(self.ui.cellNumberEdit.text(), 20, 'cell number')
        chan1 = self.ui.punctaChannelChoose[0].currentIndex()

        chan2 = self.ui.punctaChannelChoose[1].currentIndex() - 1
        chan3 = self.ui.punctaChannelChoose[2].currentIndex() - 1
        nuclear_chan = self.ui.punctaChannelChoose[3].currentIndex() - 1
        thresh = self.ui.punctaThreshText.text()
        # self.ui.disable_buttons_removeROIs()
        useDL = self.ui.useDL.isChecked()
        distance_threshold = self.ui.punctaDistText.text()
        use_analysis = self.ui.analysis_checkbox.isChecked()
        if self.ui.use_micromanager.isChecked():
            time_iterations = self.acq_settings.num_frames()
            time_interval = self.acq_settings.interval_ms() / 60000
            time_duration = time_interval * time_iterations
        else:
            time_interval = max(1, safe_int(self.ui.time_interval_text.text(), 1, 'time interval'))
            time_duration = max(time_interval, safe_int(self.ui.time_duration_text.text(), time_interval, 'time duration'))
            time_iterations = (time_duration // time_interval) + 1

        logger.info(f'start snapping setting interval {time_interval} min, iterations {time_iterations}')
        mean_capture_and_write_time = 1.2
        draw_x = self.ui.figure_x_combobox_chan.currentText() + ' ' + self.ui.figure_x_combobox_property.currentText()
        draw_y = self.ui.figure_y_combobox_chan.currentText() + ' ' + self.ui.figure_y_combobox_property.currentText()
        analysis_instruction = self.ui.get_analysis_instruction() if hasattr(self.ui, 'get_analysis_instruction') else ''
        custom_analysis_mode = self.ui.get_custom_analysis_mode() if hasattr(self.ui, 'get_custom_analysis_mode') else 'LLM Python script'
        segmentation_record = self.get_segmentation_record(mask=None)
        if len(self.cell_position_list) > 0 and not self.close:



            for t in range(time_iterations):
                logger.info(f'start imaging time point {t}')
                if self.close:
                    break
                capture_start_time = time.time()
                for i, position in enumerate(self.cell_position_list):
                    if self.close:
                        break
                    each_position_start_time = time.time()
                    self.move_microscope_absolute(position)
                    img_path = os.path.join(self.directory, self.name + '_' + str(i + 1),
                                            self.name + '_NDTiffStack.ome.tif')
                    append_imgs = []
                    for i_c, channel in enumerate(self.channels):
                        img = self.capture_image_wo_save(self.channel_group, channel, self.channel_exposures_ms[i_c])
                        append_imgs.append(img)
                    append_imgs = np.stack(append_imgs, axis=0)
                    self.snap_img_signal.emit(append_imgs.copy())
                    if use_analysis:
                        self.queue.put((append_imgs.copy(), os.path.join(os.path.dirname(img_path),
                                                                         os.path.basename(img_path).replace('.ome.tif',
                                                                                                            '_seg_record.npy')),
                                        img_path, None, (chan1, chan2, chan3, nuclear_chan),
                                        safe_float(thresh, 0.2, 'puncta threshold'),
                                        safe_int(distance_threshold, -2, 'co-distance threshold'),
                                        useDL, t, draw_x, draw_y, analysis_instruction, custom_analysis_mode,

                                        segmentation_record))



                    append_imgs = append_imgs[np.newaxis, :, :, :]
                    self.ui.writer_queue.put((append_imgs, img_path, t, time_interval))
                    capture_and_write_time = time.time() - each_position_start_time
                    logger.info(f"capture and write time: {capture_and_write_time}")
                    mean_capture_and_write_time = mean_capture_and_write_time * 0.9 + capture_and_write_time * 0.1  # momentum update mean capture and write time
                    if time_interval * 60 - (
                            time.time() - capture_start_time) < mean_capture_and_write_time + 1:  # 1s to ensure back to first position
                        warnings.warn(
                            'The time duration is supported to capture all the cells, now go back to the first cell')
                        break
                self.move_microscope_absolute(self.cell_position_list[0])
                # time.sleep(0.5)
                # self.afm_method.full_focus()
                if t < time_iterations - 1:
                    time.sleep(time_interval * 60 - (time.time() - capture_start_time))
        # self.ui.writer_queue.put((None, None, None, None))
        self.finished_signal.emit('finished')

    def cell_segmentation(self, img, file, resize=True):
        tic = time.time()
        self.ui.clear_all()
        self.flows = [[], [], []]
        if not hasattr(self.ui, 'current_model_path'):
            self.ui.initialize_model(model_name=cellpose_models.CELLQUANT_MODEL_NAME, custom=False)
        do_3D = self.ui.load_3D
        stitch_threshold = float(self.ui.stitch_threshold.text()) if not isinstance(
            self.ui.stitch_threshold, float) else self.ui.stitch_threshold
        do_3D = False if stitch_threshold > 0. else do_3D
        w, h = img.shape
        image = cv2.resize(img, None, fx=1. / self.resize_scale, fy=1. / self.resize_scale,
                           interpolation=cv2.INTER_LINEAR)
        # min_size = 0
        min_size = self.cell_min_size // (self.resize_scale ** 2)
        channels = self.ui.get_channels()
        flow_threshold, cellprob_threshold = self.ui.get_thresholds()
        self.ui.diameter = float(self.ui.Diameter.text())
        # diameter = self.ui.diameter / self.resize_scale
        diameter = self.ui.diameter
        niter = max(0, int(self.ui.niter.text()))
        niter = None if niter == 0 else niter
        normalize_params = self.ui.get_normalize_params()
        masks, flows = self.ui.model.eval(
            image, channels=channels, diameter=diameter,
            cellprob_threshold=cellprob_threshold, min_size=min_size,
            flow_threshold=flow_threshold, do_3D=do_3D, niter=niter,
            normalize=normalize_params, stitch_threshold=stitch_threshold,
        )[:2]
        masks = masks.astype(np.uint8)
        masks = cv2.resize(masks, None, fx=self.resize_scale, fy=self.resize_scale, interpolation=cv2.INTER_NEAREST)
        colored_mask = label2rgb(masks, change_to_8bit(img.copy()), bg_label=0) * 255
        if file is not None:
            cv2.imwrite(os.path.join(os.path.dirname(file), os.path.basename(file).replace('.tif', '_seg.png')),
                        colored_mask.astype(np.uint8))

        return masks, colored_mask

    def get_segmentation_record(self, mask=None):
        stitch_threshold = float(self.ui.stitch_threshold.text()) if not isinstance(
            self.ui.stitch_threshold, float) else self.ui.stitch_threshold
        normalize_params = self.ui.get_normalize_params()
        flow_threshold, cellprob_threshold = self.ui.get_thresholds()
        niter = max(0, int(self.ui.niter.text()))
        if not hasattr(self.ui, 'current_model_path'):
            self.ui.initialize_model(model_name=cellpose_models.CELLQUANT_MODEL_NAME, custom=False)
        return {'mask': mask, 'cell_channel': getattr(self, 'snap_channel_index', 0), 'use_gpu': cellpose_core.use_gpu(),
                'model_path': getattr(self.ui, 'current_model_path', None),
                'model_type': getattr(self.ui, 'current_model', None),

                'do_3D': self.ui.load_3D, 'stitch_threshold': stitch_threshold, 'resize_scale': self.resize_scale,
                'cell_min_size': self.cell_min_size, 'channels': self.ui.get_channels(),
                'flow_threshold': flow_threshold, 'cellprob_threshold': cellprob_threshold,
                'normalize_params': normalize_params, 'diameter': self.ui.diameter,
                'niter': niter}

    def save_record(self, mask, img_path):
        record = self.get_segmentation_record(mask)
        np.save(os.path.join(os.path.dirname(img_path),
                             os.path.basename(img_path).replace('.ome.tif', '_seg_record.npy')), record)


    def image_flipper(self, image, metadata):
        # if metadata['Channel'] == 'Reflected_Camera':
        #     image = np.flip(image, axis=0)
        if metadata['Reflected_FW-Label'] != 'Closed':
            image = np.flip(image, axis=0)
        self.current_image_array.append(image)
        return image, metadata

    def acquire_xyz_channel(self):
        logger.info('autofocusing')
        # self.core.full_focus()
        self.afm_method.full_focus()
        with pycromanager.Acquisition(directory=self.directory, name=self.name,
                                      image_process_fn=self.image_flipper) as acq:
            events = pycromanager.multi_d_acquisition_events(channels=self.channels,
                                                             channel_group=self.channel_group,
                                                             channel_exposures_ms=self.channel_exposures_ms,
                                                             z_start=self.z_start, z_end=self.z_end, z_step=self.z_step)
            originz = self.core.get_position()
            acq.acquire(events)

            dataset = acq.get_dataset()
            self.core.set_position(originz)

        return dataset

    def move_microscope(self, distance):
        # x>0, move to left, cell go right, y > 0. move to up, cell go down
        self.core.set_relative_xy_position(distance[0], distance[1])
        time.sleep(0.1)  # ensure move the correct position

    def move_microscope_absolute(self, position):
        currentx = self.core.get_x_position()
        currenty = self.core.get_y_position()
        distancex = abs(position[0] - currentx)
        distancey = abs(position[1] - currenty)
        times_x = distancex // 500
        times_y = distancey // 700
        self.core.set_xy_position(position[0], position[1])  # 0.1s x move 500, y move 700
        time.sleep(max(0.1 * max(times_x, times_y), 0.1))  # ensure move the correct position

    def get_imaging_settings(self):  # todo: add time lapse imaging and z-stack imaging
        if self.ui.use_micromanager.isChecked():
            self.acq_manager = self.studio.acquisitions()
            self.acq_settings = self.acq_manager.get_acquisition_settings()
            self.channels = []
            self.channel_exposures_ms = []
            self.channel_group = '3-VT-iSIM Dual Cam'
            channels = self.acq_settings.channels()
            self.usez = self.acq_settings.use_slices()
            self.z_start = self.acq_settings.slice_z_bottom_um()
            self.z_end = self.acq_settings.slice_z_top_um()
            self.z_step = self.acq_settings.slice_z_step_um()
            if channels.size() > 0:
                for i in range(channels.size()):
                    self.channels.append(channels.get(i).config())
                    self.channel_exposures_ms.append(self.acq_settings.channels().get(i).exposure())
                self.channel_group = channels.get(0).channel_group()
            else:
                raise ValueError('You must specify at least one channel and channel_group')
        else:
            # read from self-designed ui
            self.channel_group = self.ui.channel_group_down.currentText()
            self.channels = []
            self.channel_exposures_ms = []
            for i in range(4):
                if self.ui.imaging_channel_comboboxs[i].currentText() != 'None':
                    self.channels.append(self.ui.imaging_channel_comboboxs[i].currentText())
                    self.channel_exposures_ms.append(int(self.ui.imaging_exposure_comboboxs[i].text()))
            self.usez = self.ui.use_zstack.isChecked()
            self.z_start = float(self.ui.z_start_label.text())
            self.z_end = float(self.ui.z_end_label.text())
            self.z_step = float(self.ui.z_interval_text.text())
        if len(self.channels) == 0 or len(self.channel_exposures_ms) == 0:
            raise ValueError('You must specify at least one channel and channel_group')
        self.snap_channel_name = self.ui.find_cell_combox.currentText()
        self.snap_channel_index = self.channels.index(self.snap_channel_name)
        print('segmentation using', self.snap_channel_name, self.snap_channel_index)

    def get_save_path(self):
        if self.ui.use_micromanager.isChecked():
            self.directory = Path(self.acq_settings.root())
            self.name = self.acq_settings.prefix()
        else:
            self.directory = Path(self.ui.file_directory_text.text())
            self.name = self.ui.file_name_text.text()
            if self.directory is None or self.name is None:
                raise ValueError('You must specify the directory and file name')

    def capture_image_wo_save(self, channel_group, channel_name, exposure_time):
        # self.core.set_config("4-VT-iSIM Single Cam", "2-GFP")
        # self.core.set_exposure(200)
        self.core.set_config(channel_group, channel_name)
        self.core.set_exposure(exposure_time)
        # self.core.full_focus()
        self.afm_method.full_focus()
        self.core.snap_image()
        tagged_image = self.core.get_tagged_image()
        image_array = np.reshape(
            tagged_image.pix,
            newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]],
        )
        image_array, _ = self.image_flipper(image_array, tagged_image.tags)  # gfp need to flip
        return image_array

    def snap_and_analysis(self, img_path):
        width = self.width * self.pixel_size
        height = self.height * self.pixel_size
        currentx = self.core.get_x_position()
        currenty = self.core.get_y_position()
        logger.info(f'snap x and y {currentx}, {currenty}')
        new_area = box(currentx - width // 2, currenty - height // 2, currentx + width // 2, currenty + height // 2)
        overlap_ratio = self.imaged_area_tracker.check_area(new_area)
        append_imgs = []
        if overlap_ratio < self.image_overlap_ratio_thresh:
            for i_c, channel in enumerate(self.channels):
                img = self.capture_image_wo_save(self.channel_group, channel, self.channel_exposures_ms[i_c])
                append_imgs.append(img)
            image = np.stack(append_imgs, axis=0)
            imwrite(os.path.join(img_path, f'{self.snap_count}.tif'), image)
            self.snap_img_signal.emit(image)
            self.ui.loaded = True
            mask, colored_mask = self.cell_segmentation(image[self.snap_channel_index, :, :],
                                                        os.path.join(img_path, f'{self.snap_count}.tif'))
            self.snap_mask_signal.emit(mask)
            self.snapped_area_tracker.add_imaged_area(new_area)
            with open(os.path.join(self.directory, 'snap_tracker.txt'), 'a') as f:
                f.write(
                    f'{currentx - width // 2}\t{currenty - height // 2} \t'
                    f'{currentx + width // 2}\t{currenty + height // 2}')
                f.write('\n')
            height, width = mask.shape
            center_cells = []
            uncenter_cells = []  # Now stores (centeroid, cell_size, cell_id)

            for cell in range(1, np.max(mask) + 1):
                cell_mask = (mask == cell)
                cell_size = np.sum(cell_mask)

                if cell_size < self.cell_min_size:
                    mask[cell_mask] = 0
                    continue

                y_indices, x_indices = np.where(cell_mask != 0)
                centeroid = (np.mean(y_indices), np.mean(x_indices))

                if abs(centeroid[1] - width // 2) < 200 and abs(
                        centeroid[0] - height // 2) < 200 and cell_size > 300 * 300:
                    center_cells.append(cell)
                    continue

                # Store centeroid with cell size and cell id
                uncenter_cells.append((centeroid, cell_size, cell))

            return center_cells, uncenter_cells, mask, colored_mask, image
        return [], [], None, None, None

    def calculate_score(self, cell_number, cell_size, x_distance, y_distance):
        """
        Calculate movement score considering:
        - Cell size (larger cells prioritized)
        - Overlap with already imaged areas
        - Distance to move
        - Movement history to avoid oscillation
        """
        currentx = self.core.get_x_position()
        currenty = self.core.get_y_position()
        image_areax = self.width * self.pixel_size
        image_areay = self.height * self.pixel_size
        new_area = box(currentx + x_distance - image_areax // 2, currenty + y_distance - image_areay // 2,
                       currentx + x_distance + image_areax // 2, currenty + y_distance + image_areay // 2)

        # Prevent oscillation: penalize moves that are opposite to recent moves
        if hasattr(self, 'movement_history') and len(self.movement_history) >= 2:
            # Check if this move reverses the last move
            last_move = self.movement_history[-1]
            if abs(x_distance + last_move[0]) < 50 * self.pixel_size and \
                    abs(y_distance + last_move[1]) < 50 * self.pixel_size:
                return -20  # Strong penalty for reversing direction

        if cell_number > 0:  # Moving towards a specific cell
            overlap_ratio = self.imaged_area_tracker.check_area(new_area)
            if overlap_ratio >= self.image_overlap_ratio_thresh:
                score = -10  # Already imaged
            else:
                # Normalize cell size to a reasonable scale (0-1 range)
                # Assume max cell size is around 1000*1000 pixels
                normalized_cell_size = min(cell_size / (1000 * 1000), 1.0)

                # Calculate distance penalty (normalized)
                distance_penalty = (abs(x_distance) + abs(y_distance)) / (image_areax + image_areay)

                # Score formula prioritizing:
                # 1. Cell size (weight: 3.0)
                # 2. Overlap avoidance (weight: 2.0)
                # 3. Distance minimization (weight: -1.0)
                score = (3.0 * normalized_cell_size) + \
                        (2.0 * (1 - overlap_ratio)) - \
                        (1.0 * distance_penalty)
        else:  # Exploratory move
            overlap_ratio = self.snapped_area_tracker.check_area(new_area)
            score = (1 - overlap_ratio) * 0.5  # Lower score for exploration

        return score

    def decide_how_to_move(self, uncenter_cells):
        """
        Decide next movement based on:
        1. Prioritizing larger cells
        2. Avoiding already imaged areas
        3. Preventing oscillation
        """
        # Initialize movement history if not exists
        if not hasattr(self, 'movement_history'):
            self.movement_history = []

        distances = []
        scores = []
        directions = []
        cell_info = []  # Store cell information for logging

        # Sort uncenter_cells by size (largest first)
        uncenter_cells_sorted = sorted(uncenter_cells, key=lambda x: x[1], reverse=True)

        # Consider moves towards cells (prioritize larger cells)
        for centeroid, cell_size, cell_id in uncenter_cells_sorted:
            x_distance = -(centeroid[1] - self.width // 2) * self.pixel_size
            y_distance = (self.height // 2 - centeroid[0]) * self.pixel_size

            distances.append([y_distance, x_distance])
            scores.append(self.calculate_score(1, cell_size, x_distance, y_distance))
            cell_info.append(f"cell_{cell_id}_size_{cell_size}")

            if x_distance >= 0 and y_distance >= 0:
                directions.append('upper right cell')
            elif x_distance >= 0 and y_distance <= 0:
                directions.append('low right cell')
            elif x_distance <= 0 and y_distance <= 0:
                directions.append('low left cell')
            else:
                directions.append('upper left cell')

        # Add exploratory moves (full field movements)
        exploratory_moves = []

        if self.last_movement != 'lower':
            exploratory_moves.append(('upper', 0, self.height))
        if self.last_movement != 'upper':
            exploratory_moves.append(('lower', 0, -self.height))
        if self.last_movement != 'right':
            exploratory_moves.append(('left', self.width, 0))
        if self.last_movement != 'left':
            exploratory_moves.append(('right', -self.width, 0))

        for direction, x_move, y_move in exploratory_moves:
            x_distance = x_move * self.pixel_size
            y_distance = y_move * self.pixel_size
            distances.append([y_distance, x_distance])
            scores.append(self.calculate_score(0, 0, x_distance, y_distance))
            directions.append(direction)
            cell_info.append('exploration')

        # Find best move
        max_score = max(scores)

        # If all scores are negative, choose the least negative exploration move
        if max_score <= 0:
            logger.info(f"All moves have negative scores, choosing best exploration")
            # Filter to only exploration moves
            # exploration_indices = [i for i, info in enumerate(cell_info) if info == 'exploration']
            # if exploration_indices:
            #     exploration_scores = [scores[i] for i in exploration_indices]
            #     best_exploration_idx = exploration_indices[exploration_scores.index(max(exploration_scores))]
            #     score_index = best_exploration_idx
            # else:
            #     score_index = scores.index(max_score)
        # else:
        score_index = scores.index(max_score)

        # Update movement history (keep last 3 moves)
        selected_distance = distances[score_index]
        self.movement_history.append((selected_distance[1], selected_distance[0]))  # (x, y)
        if len(self.movement_history) > 3:
            self.movement_history.pop(0)

        logger.info(f'{self.snap_count} move to {directions[score_index]}, score: {max_score:.3f}')
        logger.info(f'All scores: {[f"{d}:{s:.2f}({c})" for d, s, c in zip(directions, scores, cell_info)]}')

        self.last_movement = directions[score_index]
        self.last_ymove = selected_distance[0]
        self.last_xmove = selected_distance[1]

        return selected_distance
# if __name__ == '__main__':
#     sm = SmartMicroscope()
#     sm.imaging()
