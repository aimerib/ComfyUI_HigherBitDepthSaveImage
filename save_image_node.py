from PIL import Image, ImageCms
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
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix of the filename of the image you want to save."}),
                "bit_depth": (["8", "16", "32"], {"default": "8", "tooltip": "The bit depth of the image you want to save."}),
                "format": (["PNG", "TIFF", "JPEG"], {"default": "PNG", "tooltip": "The format of the image you want to save."}),
            },
            "optional": {
                "new_icc_profile": (["LAB", "XYZ", "sRGB", "Default"], {"default": "Default", "tooltip": "The ICC profile to embed in the saved image.\n'LAB', 'XYZ', and 'sRGB' are color spaces.\n'LAB' is a color space that is designed to be perceptually uniform, meaning that the color difference between two points in the color space is proportional to the actual color difference perceived by the human eye.\n'XYZ' is a color space that is used in the CIE standard for colorimetry.\n'sRGB' is a color space that is widely used in digital imaging and is supported by most web browsers and operating systems."}),
                "existing_icc_profile_path": ("STRING", {"default": "", "tooltip": "The custom ICC profile to embed in the saved image.\nThe path to the ICC profile file."})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Save image with higher bit depth and format."

    def save_images(self, images, bit_depth, new_icc_profile="Default", existing_icc_profile_path="", format="PNG", filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        try:
            if new_icc_profile != "Default" and existing_icc_profile_path != "":
                raise ValueError("Cannot use both new_icc_profile and existing_icc_profile_path. Please provide only one.")

            filename_prefix += self.prefix_append
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
            results = list()

            for (batch_number, image) in enumerate(images):
                print(f"Processing image {batch_number + 1} of {len(images)}")

                img_np = image.cpu().numpy()

                print(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}")
                print(f"Image min: {np.min(img_np)}, max: {np.max(img_np)}")

                log_channel_stats(img_np, "Input")
                save_debug_image(img_np, "input", batch_number, full_output_folder)


                if img_np.ndim != 3 or img_np.shape[2] != 3:
                    raise ValueError(f"Expected an RGB image with 3 channels, got shape {img_np.shape}")

                # Normalize input if necessary
                if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                    if np.min(img_np) < 0 or np.max(img_np) > 1:
                        print("Input values out of expected range [0, 1], normalizing")
                        img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))

                log_channel_stats(img_np, "After normalization")
                save_debug_image(img_np, "normalized", batch_number, full_output_folder)

                # Convert to appropriate bit depth
                if bit_depth == "32":
                    format = "TIFF"
                    # img_np = (img_np * 65535).astype(np.uint16)
                    if img_np.max() > 1:  # Check if the input is in [0, 255] range
                        print("Input detected in [0, 255] range. Normalizing to [0, 1].")
                        img_np = img_np / 255.0
                    else:
                        print("Input detected in [0, 1] range. No normalization needed.")

                    img_np = img_np.astype(np.float32)
                elif bit_depth == "16":
                    format = "TIFF" if format == "TIFF" else "PNG"
                    if img_np.max() > 1:
                        print("Input detected in [0, 255] range. Normalizing to [0, 1].")
                        img_np = img_np / 255.0
                    else:
                        print("Input detected in [0, 1] range. No normalization needed.")

                    img_np = np.clip(img_np * 65535.0, 0, 65535).astype(np.uint16)
                else:  # 8-bit
                    if img_np.dtype == np.uint8 and img_np.max() <= 255 and img_np.min() >= 0:
                        print("Input is already in 8-bit format. No conversion needed.")
                    elif img_np.max() <= 1.0 and img_np.min() >= 0:
                        print("Input detected in [0, 1] float range. Converting to [0, 255] uint8.")
                        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
                    else:
                        print("Input in non-standard range. Normalizing to [0, 255] uint8.")
                        img_np = np.clip((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255.0, 0, 255).astype(np.uint8)
                    # img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

                print(f"Converted image shape: {img_np.shape}, dtype: {img_np.dtype}")
                print(f"Converted image min: {np.min(img_np)}, max: {np.max(img_np)}")

                log_channel_stats(img_np, "After bit depth conversion")
                save_debug_image(img_np, "bit_depth_converted", batch_number, full_output_folder)

                if img_np[:,:,0].mean() < img_np[:,:,2].mean():
                    print("Detected BGR format, converting to RGB")
                    return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

                img = Image.fromarray(img_np, mode='RGB')

                # Handle ICC profile
                if new_icc_profile != "Default":
                    icc_profile_cms_obj = ImageCms.createProfile(new_icc_profile)
                    icc_profile_path = ImageCms.getProfileName(icc_profile_cms_obj)
                    icc_profile = ImageCms.getOpenProfile(icc_profile_path)
                    img.info['icc_profile'] = icc_profile.tobytes()
                elif existing_icc_profile_path:
                    icc_profile = ImageCms.getOpenProfile(existing_icc_profile_path)
                    img.info['icc_profile'] = icc_profile.tobytes()

                # Save the image
                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                log_channel_stats(img_np, "Before final save")
                save_debug_image(img_np, "final", batch_number, full_output_folder)
                file = self.save_image(img, format, bit_depth, full_output_folder, filename_with_batch_num, counter, prompt, extra_pnginfo)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
                counter += 1

            return { "ui": { "images": results } }

        except Exception as e:
            print(f"An error occurred while saving images: {str(e)}")
            raise

    def save_image(self, img, format, bit_depth, full_output_folder, filename_with_batch_num, counter, metadata=None, icc_profile=None):
        extensions = {"JPEG": "jpg", "PNG": "png", "TIFF": "tiff"}
        file = f"{filename_with_batch_num}_{counter:05}_.{extensions[format]}"
        full_path = os.path.join(full_output_folder, file)

        save_kwargs = {}
        if icc_profile:
            save_kwargs['icc_profile'] = img.info.get('icc_profile')

        if format == "PNG":
            if bit_depth == "8":
                img.save(full_path, format="PNG", pnginfo=metadata, compress_level=self.compress_level, **save_kwargs)
            else:
                img.save(full_path, format="PNG", **save_kwargs)
        elif format == "JPEG":
            img.save(full_path, format="JPEG", **save_kwargs)
        elif format == "TIFF":
            img.save(full_path, format="TIFF", **save_kwargs)

        return file

def save_debug_image(img_np, stage, batch_number, full_output_folder):
    debug_img = Image.fromarray((img_np * 255).astype(np.uint8) if img_np.dtype != np.uint8 else img_np)
    debug_filename = f"debug_{stage}_batch{batch_number}.png"
    debug_path = os.path.join(full_output_folder, debug_filename)
    debug_img.save(debug_path)
    print(f"Saved debug image for {stage}: {debug_path}")

def log_channel_stats(img_np, stage):
    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = img_np[:,:,i]
        print(f"{stage} - {channel} channel: min={np.min(channel_data):.4f}, max={np.max(channel_data):.4f}, mean={np.mean(channel_data):.4f}")
