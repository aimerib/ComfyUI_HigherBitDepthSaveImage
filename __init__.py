from .save_image_node import SaveImageHigherBitDepth

NODE_CLASS_MAPPINGS = {
    "SaveImageHigherBitDepth": SaveImageHigherBitDepth,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageHigherBitDepth": "Save Image Higher Bit Depth",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
