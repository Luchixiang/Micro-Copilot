from tifffile import imread,imwrite
from multiprocessing import Process
import numpy as np
class WriterProcess(Process):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        while True:
            image, img_path, t, time_interval = self.queue.get()
            if image is None:
                break
            if t > 0:
                origin_img = imread(img_path)
                if origin_img.ndim == 3:
                    origin_img = origin_img[np.newaxis, :, :, :]
                elif origin_img.ndim == 2:
                    origin_img = origin_img[np.newaxis, :, :]
                image = np.concatenate([origin_img, image], axis=0)
            try:
                imwrite(img_path, image, metadata={'axes': 'TCYX', 'TimeIncrement': time_interval * 60})
            except:
                print(f'writing {img_path} filed')
