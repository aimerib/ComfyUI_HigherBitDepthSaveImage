from PIL import Image, ImageCms
import numpy as np
import cv2
import folder_paths
from PIL.PngImagePlugin import PngInfo
import os
import json
import tifffile
import png  # pypng
import imageio
import zlib
import piexif  # For JPEG EXIF metadata

# from comfy.cli_args import args
# import comfy.utils

def get_profile_description(icc_profile):
    """
    Extracts the ICC profile description.

    Args:
        icc_profile (ImageCmsProfile): The opened ICC profile.

    Returns:
        str: The profile description or 'sRGB' if extraction fails.
    """
    try:
        # Access the 'description' attribute from the profile
        description = icc_profile.profile.profile_description
        # Ensure the profile name is within PNG's limitations (1-79 bytes)
        if len(description.encode('latin1')) > 79:
            print("Profile description too long for PNG iCCP chunk. Defaulting to 'sRGB'.")
            return 'sRGB'
        return description
    except Exception as e:
        print(f"Could not extract profile description: {e}. Defaulting to 'sRGB'.")
        return 'sRGB'

def create_iccp_chunk(icc_profile_bytes, profile_name='sRGB'):
    """
    Creates the iCCP chunk data required for embedding ICC profiles in PNGs.

    Args:
        icc_profile_bytes (bytes): Raw ICC profile data.
        profile_name (str): Name of the ICC profile. Must be 1-79 bytes.

    Returns:
        tuple: A tuple containing the chunk type and data.
    """
    if len(profile_name.encode('latin1')) > 79:
        raise ValueError("Profile name too long for iCCP chunk (max 79 bytes).")

    # Encode profile name and ensure it's null-terminated
    profile_name_encoded = profile_name.encode('latin1') + b'\x00'

    # Compression method is 0 (deflate)
    compression_method = b'\x00'

    # Compress the ICC profile data using zlib
    compressed_icc = zlib.compress(icc_profile_bytes)

    # Combine to form the iCCP chunk data
    iccp_data = profile_name_encoded + compression_method + compressed_icc

    return ('iCCP', iccp_data)

def save_png_with_icc(full_path, img_np, icc_profile_bytes, profile_name='sRGB', metadata_dict=None):
    """
    Saves a 16-bit PNG with an embedded ICC profile and additional metadata using pypng.

    Args:
        full_path (str): Path to save the PNG file.
        img_np (numpy.ndarray): Image data as a NumPy array with dtype uint16.
        icc_profile_bytes (bytes): Raw ICC profile data.
        profile_name (str): Name of the ICC profile.
        metadata_dict (dict, optional): Additional metadata to embed.

    Raises:
        ValueError: If the image data is not in the correct format.
    """
    if img_np.dtype != np.uint16:
        print(f"Image data dtype: {img_np.dtype}")
        raise ValueError("Image data must be uint16 for 16-bit PNGs.")
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        raise ValueError("Image data must have 3 channels (RGB).")

    height, width, channels = img_np.shape
    # Reshape the image to a list of rows
    rows = img_np.reshape(-1, channels).tolist()

    # Create the iCCP chunk
    chunks = []
    if icc_profile_bytes:
        iccp_chunk = create_iccp_chunk(icc_profile_bytes, profile_name)
        chunks.append(iccp_chunk)

    # Add additional metadata as text chunks
    if metadata_dict:
        for key, value in metadata_dict.items():
            # Each key-value pair is added as a separate text chunk
            text_chunk = ('tEXt', f"{key}\x00{value}".encode('latin1'))
            chunks.append(text_chunk)

    # Initialize the PNG writer
    writer = png.Writer(width=width, height=height, bitdepth=16, compression=9, greyscale=False)

    # Write the PNG file with the iCCP and additional metadata chunks
    with open(full_path, 'wb') as f:
        writer.write(f, rows, chunks=chunks)

    print(f"Saved 16-bit PNG with ICC profile and metadata at: {full_path}")


def embed_metadata_tiff(full_path, img_np, metadata_dict):
    """
    Embeds metadata into a TIFF file using tifffile.

    Args:
        full_path (str): Path to save the TIFF file.
        img_np (numpy.ndarray): Image data as a NumPy array.
        metadata_dict (dict): Dictionary of metadata to embed.

    Raises:
        ValueError: If metadata_dict is not a dictionary.
    """
    if not isinstance(metadata_dict, dict):
        raise ValueError("Metadata must be a dictionary.")

    # Convert metadata_dict to JSON string
    metadata_json = json.dumps(metadata_dict)

    # Prepare the metadata
    metadata = {'Software': 'ComfyUI', 'Description': metadata_json}

    # Save using tifffile with metadata
    tifffile.imwrite(
        full_path,
        img_np,
        photometric='rgb',
        metadata=metadata,
    )

    print(f"Saved TIFF with embedded metadata at: {full_path}")

def embed_metadata_jpeg(full_path, img_pil, metadata_dict):
    """
    Embeds metadata into a JPEG file using piexif.

    Args:
        full_path (str): Path to save the JPEG file.
        img_pil (PIL.Image.Image): PIL Image object.
        metadata_dict (dict): Dictionary of metadata to embed.

    Raises:
        ValueError: If metadata_dict is not a dictionary.
    """
    if not isinstance(metadata_dict, dict):
        raise ValueError("Metadata must be a dictionary.")

    # Convert metadata_dict to JSON string
    metadata_json = json.dumps(metadata_dict)

    # Create EXIF data
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    # Embed the JSON metadata in the UserComment tag (Exif tag 37510)
    # UserComment requires a specific format: 8-byte prefix + comment
    prefix = b"ComfyUI\x00\x00\x00"  # Custom prefix to identify the source
    user_comment = prefix + metadata_json.encode('utf-8')
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment

    # Convert exif_dict to bytes
    exif_bytes = piexif.dump(exif_dict)

    # Save the JPEG with EXIF data
    img_pil.save(full_path, "JPEG", exif=exif_bytes, quality=95)

    print(f"Saved JPEG with embedded metadata at: {full_path}")

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
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
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
            # Adjust the order of dimensions based on assumption
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
            # if img_shape[0] == 3:
            #     # Likely (channels, height, width)
            # elif img_shape[2] == 3:
            #     # Already (height, width, channels)
            #     full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            #         filename_prefix, self.output_dir, img_shape[1], img_shape[0]
            #     )
            # else:
            #     raise ValueError(f"Unexpected image shape: {img_shape}")

            results = list()

            for (batch_number, image) in enumerate(images):
                print(f"Processing image {batch_number + 1} of {len(images)}")

                img_np = image.cpu().numpy()

                # Detect and transpose if necessary
                if img_np.shape[0] == 3:
                    img_np = np.transpose(img_np, (1, 2, 0))  # Transpose to (height, width, channels)
                    print("Transposed image from (channels, height, width) to (height, width, channels).")
                elif img_np.shape[2] == 3:
                    pass  # Already in (height, width, channels)
                else:
                    raise ValueError(f"Unexpected image shape after conversion: {img_np.shape}")

                print(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}")
                print(f"Image min: {np.min(img_np)}, max: {np.max(img_np)}")

                log_channel_stats(img_np, "Input")
                save_debug_image(img_np, "input", counter, full_output_folder)

                if img_np.ndim != 3 or img_np.shape[2] != 3:
                    raise ValueError(f"Expected an RGB image with 3 channels, got shape {img_np.shape}")


                # Normalize input if necessary
                if img_np.dtype in [np.float32, np.float64]:
                    if np.min(img_np) < 0 or np.max(img_np) > 1:
                        print("Input values out of expected range [0, 1], normalizing")
                        img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))

                log_channel_stats(img_np, "After normalization")
                save_debug_image(img_np, "normalized", counter, full_output_folder)

                # Initialize ICC profile bytes and profile name
                icc_profile_bytes = None
                profile_name = 'sRGB'  # Default profile name

                # Handle ICC profile
                if new_icc_profile != "Default":
                    icc_profile_cms_obj = ImageCms.createProfile(new_icc_profile)
                    icc_profile_path = ImageCms.getProfileName(icc_profile_cms_obj)
                    icc_profile = ImageCms.getOpenProfile(icc_profile_cms_obj)
                    icc_profile_bytes = icc_profile.tobytes()
                    profile_name = get_profile_description(icc_profile)
                elif existing_icc_profile_path:
                    icc_profile = ImageCms.getOpenProfile(existing_icc_profile_path)
                    icc_profile_bytes = icc_profile.tobytes()
                    profile_name = get_profile_description(icc_profile)

                # Convert to appropriate bit depth
                if bit_depth == "32":
                    format = "TIFF"  # TIFF supports higher bit depths
                    # Ensure the data is in float32
                    if img_np.max() > 1.0:  # Check if the input is in [0, 255] range
                        print("Input detected in [0, 255] range. Normalizing to [0, 1].")
                        img_np = img_np / 255.0
                    else:
                        print("Input detected in [0, 1] range. No normalization needed.")

                    img_np = img_np.astype(np.float32)
                elif bit_depth == "16":
                    format = "TIFF" if format == "TIFF" else "PNG"
                    if img_np.max() <= 1.0:
                        print("Input detected in [0, 1] range. Scaling to [0, 65535].")
                        img_np = img_np * 65535.0
                    else:
                        print("Input detected in [0, 255] range. Scaling to [0, 65535].")
                        img_np = img_np * (65535.0 / 255.0)

                    img_np = np.clip(img_np, 0, 65535).astype(np.uint16)
                else:  # 8-bit
                    if img_np.dtype == np.uint8 and img_np.max() <= 255 and img_np.min() >= 0:
                        print("Input is already in 8-bit format. No conversion needed.")
                    elif img_np.max() <= 1.0 and img_np.min() >= 0:
                        print("Input detected in [0, 1] float range. Converting to [0, 255] uint8.")
                        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
                    else:
                        print("Input in non-standard range. Normalizing to [0, 255] uint8.")
                        img_np = np.clip((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255.0, 0, 255).astype(np.uint8)

                print(f"Converted image shape: {img_np.shape}, dtype: {img_np.dtype}")

                log_channel_stats(img_np, "After bit depth conversion")
                save_debug_image(img_np, "bit_depth_converted", counter, full_output_folder)

                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))

                # Log before saving
                log_channel_stats(img_np, "Before final save")
                save_debug_image(img_np, "final", counter, full_output_folder)

                # Prepare metadata for embedding
                metadata = None
                # if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                metadata_dict = {}
                if metadata:
                    metadata_dict = extract_metadata_from_pnginfo(metadata)

                # Save the image
                file = self.save_image(
                    img_np,
                    format,
                    bit_depth,
                    full_output_folder,
                    filename_with_batch_num,
                    counter,
                    metadata,
                    icc_profile_bytes=icc_profile_bytes,
                    profile_name=profile_name,
                    metadata_dict=metadata_dict  # Pass the metadata dictionary
                )
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

    def save_image(self, img_np, format, bit_depth, full_output_folder, filename_with_batch_num, counter, extra_pnginfo=None, icc_profile_bytes=None, profile_name='sRGB', metadata_dict=None):
        """
        Saves the image in the specified format and bit depth, embedding ICC profiles and metadata as needed.

        Args:
            img (PIL.Image.Image): The PIL Image to save.
            format (str): The image format ('PNG', 'TIFF', 'JPEG').
            bit_depth (str): The bit depth ('8', '16', '32').
            full_output_folder (str): The directory to save the image.
            filename_with_batch_num (str): The filename with batch number.
            counter (int): The image counter.
            prompt (str, optional): Additional prompt information.
            extra_pnginfo (PngInfo, optional): Extra PNG metadata.
            icc_profile_bytes (bytes, optional): ICC profile bytes.
            profile_name (str, optional): Name of the ICC profile.
            metadata_dict (dict, optional): Metadata to embed in all formats.

        Returns:
            str: The saved filename.
        """
        extensions = {"JPEG": "jpg", "PNG": "png", "TIFF": "tiff"}
        file = f"{filename_with_batch_num}.{extensions[format]}"
        full_path = os.path.join(full_output_folder, file)

        print(f"Saving file: {full_path}")
        print(f"Format: {format}, Bit Depth: {bit_depth}, ICC Profile: {'Yes' if icc_profile_bytes else 'No'}, Profile Name: {profile_name}")

        img = Image.fromarray(img_np, mode='RGB')

        if format == "TIFF":
            # Handle metadata embedding for TIFF
            if metadata_dict:
                embed_metadata_tiff(full_path, img_np, metadata_dict)
            else:
                # Save without additional metadata
                tifffile.imwrite(
                    full_path,
                    img_np,
                    photometric='rgb',
                    metadata=None,
                )
                print(f"Saved TIFF without additional metadata at: {full_path}")
        elif format == "PNG" and bit_depth == "16":
            # Use pypng to save 16-bit PNG with ICC Profile and metadata
            if img_np.dtype != np.uint16:
                raise ValueError("Image data must be uint16 for 16-bit PNGs.")
            if img_np.ndim != 3 or img_np.shape[2] != 3:
                raise ValueError("Image data must have 3 channels (RGB).")
            if icc_profile_bytes:
                # Create a PIL Image from the NumPy array
                img.save(
                    full_path,
                    format='PNG',
                    pnginfo=extra_pnginfo,
                    icc_profile=icc_profile_bytes,
                    compress_level=self.compress_level  # Adjust compression level as needed
                )
            else:
                # Save without ICC Profile and metadata
                img.save(full_path, format="PNG", pnginfo=extra_pnginfo, compress_level=self.compress_level)
                print(f"Saved 16-bit PNG without ICC profile: {full_path}")
        elif format == "PNG" and bit_depth == "8":
            # Save 8-bit PNG with extra_pnginfo
            img.save(full_path, format="PNG", pnginfo=extra_pnginfo, compress_level=self.compress_level)
            print(f"Saved 8-bit PNG at: {full_path}")
        elif format == "JPEG":
            # Handle metadata embedding for JPEG
            if metadata_dict:
                embed_metadata_jpeg(full_path, img, metadata_dict)
            else:
                img.save(full_path, format="JPEG", quality=95)
                print(f"Saved JPEG without additional metadata at: {full_path}")
        else:
            # Handle other formats and bit depths with PIL
            save_kwargs = {}
            if icc_profile_bytes:
                save_kwargs['icc_profile'] = icc_profile_bytes
                print("ICC Profile embedded using PIL.")

            img.save(full_path, format=format, **save_kwargs)
            print(f"Saved {format} at: {full_path}")

        return file

def save_debug_image(img_np, stage, counter, full_output_folder):
    """
    Saves a debug image for the specified stage.

    Args:
        img_np (numpy.ndarray): Image data as a NumPy array.
        stage (str): The processing stage.
        counter (int): The image counter.
        full_output_folder (str): The directory to save the debug image.
    """
    if img_np.dtype in [np.float32, np.float64]:
        if img_np.max() <= 1.0:
            debug_img = Image.fromarray((img_np * 255).astype(np.uint8))
        else:
            debug_img = Image.fromarray((np.clip(img_np / np.max(img_np), 0, 1) * 255).astype(np.uint8))
    elif img_np.dtype == np.uint16:
        debug_img = Image.fromarray((img_np / 257).astype(np.uint8))  # Scale down to 8-bit
    else:
        debug_img = Image.fromarray(img_np.astype(np.uint8))

    debug_filename = f"debug_{stage}_batch{counter}.png"
    debug_path = os.path.join(full_output_folder, debug_filename)
    debug_img.save(debug_path)
    print(f"Saved debug image for {stage}: {debug_path}")


def log_channel_stats(img_np, stage):
    """
    Logs statistics for each color channel.

    Args:
        img_np (numpy.ndarray): Image data as a NumPy array.
        stage (str): The processing stage.
    """
    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = img_np[:,:,i]
        print(f"{stage} - {channel} channel: min={np.min(channel_data):.4f}, max={np.max(channel_data):.4f}, mean={np.mean(channel_data):.4f}")

def extract_metadata_from_pnginfo(pnginfo: PngInfo) -> dict:
    """
    Extracts metadata key-value pairs from a PngInfo object.

    Args:
        pnginfo (PngInfo): The PngInfo object containing metadata.

    Returns:
        dict: A dictionary of metadata key-value pairs.
    """
    metadata = {}
    for chunk_type, data, _ in pnginfo.chunks:
        if chunk_type == b'tEXt':
            try:
                key, value = data.split(b'\x00', 1)
                key = key.decode('latin-1')
                value = value.decode('latin-1')
                metadata[key] = value
            except ValueError:
                print("Invalid tEXt chunk format.")
        elif chunk_type == b'zTXt':
            try:
                key, compressed_value = data.split(b'\x00', 1)
                key = key.decode('latin-1')
                value = zlib.decompress(compressed_value).decode('latin-1')
                metadata[key] = value
            except (ValueError, zlib.error):
                print("Invalid zTXt chunk format or compression.")
        elif chunk_type == b'iTXt':
            try:
                # iTXt format: key\0compression_flag\0language\0translated_keyword\0text
                parts = data.split(b'\x00', 4)
                if len(parts) >= 5:
                    key, compression_flag, lang, tkey, text = parts
                    key = key.decode('latin-1')
                    if compression_flag == b'\x01':
                        # Text is compressed
                        value = zlib.decompress(text).decode('utf-8')
                    else:
                        value = text.decode('utf-8')
                    metadata[key] = value
            except (ValueError, zlib.error):
                print("Invalid iTXt chunk format or compression.")
    return metadata
