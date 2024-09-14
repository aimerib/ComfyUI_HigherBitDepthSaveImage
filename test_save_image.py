# test_save_image.py
# Test the save_image_node.py script

import torch
from PIL.PngImagePlugin import PngInfo
from save_image_node import SaveImageHigherBitDepth

def main():
    # Instantiate the class
    test = SaveImageHigherBitDepth()

    # Create a synthetic image tensor (RGB, 256x256) with values in [0,1]
    img_tensor = torch.rand(3, 256, 256)

    # Create extra_pnginfo with metadata
    png_info = {}
    png_info['Author'] = 'TestUser'
    png_info['Description'] = 'Synthetic Image for Testing'

    # Call the save_images method
    test.save_images(
        images=[img_tensor],
        bit_depth='16',
        format='PNG',
        filename_prefix='TestImage',
        extra_pnginfo=png_info,
        new_icc_profile='XYZ',
        # existing_icc_profile_path="/Users/aimeri/Downloads/Adobe ICC Profiles (end-user)/RGB/AdobeRGB1998.icc"
    )

if __name__ == "__main__":
    main()
