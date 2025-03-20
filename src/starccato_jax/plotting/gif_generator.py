import glob

from natsort import natsorted
from PIL import Image

from ..logging import logger


def generate_gif(
    image_pattern: str,
    output_gif: str,
    duration: float = 100,
    final_pause: float = 500,
):
    """
    Generate a looping GIF from PNG images matching a regex pattern.

    Example usage:
    generate_gif("frames/*.png", "output.gif", duration=100, final_pause=500)

    Parameters:
    - image_pattern (str): Regex pattern to match PNG images.
    - output_gif (str): Filename for the output GIF.
    - duration (float): Duration of each frame in milliseconds.
    - final_pause (float): Extra time for the last frame before looping.
    """
    # Get all matching image filenames
    image_files = glob.glob(image_pattern)

    logger.debug(
        f"Generating GIF from {image_pattern}-->{len(image_files)} images: {image_files}"
    )

    # Sort numerically based on numbers in filenames
    image_files = natsorted(image_files)

    if not image_files:
        logger.error("No matching images found!")
        return

    # Load images
    images = [Image.open(img) for img in image_files]

    # Add extra time to the last frame
    durations = [duration] * (len(images) - 1) + [final_pause]

    # Save as GIF
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,  # Loop forever
    )

    logger.info(f"GIF saved as {output_gif}")
