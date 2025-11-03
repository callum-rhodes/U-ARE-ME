import cv2
import os
####################################################################################
# Output writer                                                                   #
####################################################################################
class OutputWriter:
    """
    Utility for writing outputs either as an MP4 video (via successive calls to
    write) or as a single image file when initialized with a .png path.

    Example usage:
        output_writer = OutputWriter('output.mp4', fps=30)
        for frame in frames:
            output_writer.write(frame)  # frame is a RGB numpy array (H, W, 3)
        output_writer.close()

        output_writer = OutputWriter('output.png')
        output_writer.write(image)  # Writes/overwrites output.png
    """

    def __init__(self, output_path, fps=30):
        self.output_path = output_path
        self.fps = fps
        self._video_writer = None
        self._mode = None
        self.written = False
        self.temp_file_path = 'temp.mp4'
        if output_path is None:
            return
        out_lower = (output_path or '').lower()
        if out_lower.endswith('.mp4'):
            self._mode = 'video'
        elif out_lower.endswith('.png'):
            self._mode = 'image'
        else:
            raise ValueError("OutputWriter only supports '.mp4' or '.png' outputs")

    def write(self, img):
        """
        Writes a frame to the MP4 or an image to disk.

        - For video: initializes writer on first call using frame size.
        - For image (.png): writes/overwrites the file on every call.
        """
        if self._mode is None:
            return
        if self._mode == 'image':
            cv2.imwrite(self.output_path, img)
            self.written = True
            return

        # Video path
        if self._video_writer is None:
            if img is None:
                raise ValueError('First frame is None; cannot initialize video writer')
            if img.ndim < 2:
                raise ValueError('Invalid frame shape; expected HxW or HxWxC')

            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._video_writer = cv2.VideoWriter(self.temp_file_path, fourcc, float(self.fps), (w, h))
            if not self._video_writer.isOpened():
                self._video_writer = None
                raise RuntimeError(f'Failed to open video writer for {self.output_path}')

        self._video_writer.write(img)
        self.written = True

    def close(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            os.system(f"ffmpeg -y -loglevel quiet -i {self.temp_file_path} -map 0 -c:v libx264 -c:a copy -crf 23 -preset fast {self.output_path}")
            os.remove(self.temp_file_path)
        if self.written:
            print(f"Written output to {self.output_path}")
            self.written = False

    def __del__(self):
        # Best-effort cleanup
        try:
            self.close()
        except Exception:
            pass