from PIL import Image
import numpy as np
import cv2
import folder_paths
from PIL.PngImagePlugin import PngInfo
import os
import json

from comfy.cli_args import args
import comfy.utils

class SaveImageHigherBitDepth:
    def __init__(self):
      self.output_dir = folder_paths.get_output_directory()
      self.type = "output"
      self.prefix_append = ""
      self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix of the filename of the image you want to save."}),
                "bit_depth": (["8", "16", "32"], {"default": "8", "tooltip": "The bit depth of the image you want to save."}),
                "format": (["PNG", "TIFF", "JPEG"], {"default": "PNG", "tooltip": "The format of the image you want to save."})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Save image with higher bit depth and format."

    def save_images(self, images, bit_depth, format="PNG", filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            img_np = image.cpu().numpy()

            if img_np.ndim != 3 or img_np.shape[2] != 3:
                raise ValueError("Expected an RGB image with 3 channels.")

            if bit_depth == "32":
                format = "TIFF"
                img = img_np.astype(np.float32)
                # img = self.convert_opencv_to_pil(img_np)
            elif bit_depth == "16":
                format = "TIFF" if format == "TIFF" else "PNG"
                img = np.clip(img_np * 65535.0, 0, 65535).astype(np.uint16)
                # img = self.convert_opencv_to_pil(img_np)
            else:
                i = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
                img = Image.fromarray(i, mode='RGB')
            metadata = None
            if not args.disable_metadata and format == "PNG":
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            # file = f"{filename_with_batch_num}_{counter:05}_.png"
            # img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            if format == "JPEG":
                file = f"{filename_with_batch_num}_{counter:05}_.jpg"
                img.save(os.path.join(full_output_folder, file))
            elif format == "PNG" and bit_depth == "8":
                file = f"{filename_with_batch_num}_{counter:05}_.png"
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            elif format == "PNG":
                file = f"{filename_with_batch_num}_{counter:05}_.png"
                cv2.imwrite(os.path.join(full_output_folder, file), img)
                # img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            elif format == "TIFF":
                file = f"{filename_with_batch_num}_{counter:05}_.tiff"
                cv2.imwrite(os.path.join(full_output_folder, file), img)
                # img.save(os.path.join(full_output_folder, file))
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

    def convert_opencv_to_pil(self, img_np):
        try:
            if img_np.dtype == np.float32:
                # For 32-bit float images
                img_np = (img_np * 65535).astype(np.uint16)
            elif img_np.dtype == np.uint8:
                # For 8-bit images, no change needed
                pass
            elif img_np.dtype == np.uint16:
                # For 16-bit images, no change needed
                pass
            else:
                raise ValueError(f"Unsupported image dtype: {img_np.dtype}")

            # Convert BGR to RGB
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            # Create PIL Image directly from numpy array
            img = Image.fromarray(img_np)

            return img

        except Exception as e:
            print(f"Error converting OpenCV image to PIL: {e}")
            raise
