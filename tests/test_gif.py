import os
from tempfile import TemporaryDirectory

import pytest
from PIL import Image

from starccato_jax.plotting.gif_generator import (
    generate_gif,  # Import your function
)


@pytest.fixture
def create_test_images(outdir):
    """Fixture to create temporary test images."""

    # Create a temporary directory to store test images
    temp_dir = f"{outdir}/tmp"
    os.makedirs(temp_dir, exist_ok=True)

    test_images = [
        os.path.join(temp_dir, f"frame_{i}.png") for i in range(1, 6)
    ]

    # Create simple test images
    for i, img_path in enumerate(test_images):
        img = Image.new("RGB", (100, 100), color=(i * 40, i * 20, i * 10))
        img.save(img_path)

    yield temp_dir, test_images  # Provide test data to the test function

    # Clean up
    for img_path in test_images:
        os.remove(img_path)
    os.rmdir(temp_dir)


@pytest.mark.parametrize("duration, final_pause", [(100, 500), (50, 200)])
def test_generate_gif(create_test_images, duration, final_pause, outdir):
    """Test the GIF generation function."""
    temp_dir, test_images = create_test_images
    output_gif = os.path.join(outdir, "output.gif")

    # Call function
    generate_gif(
        os.path.join(temp_dir, "frame_*.png"),
        output_gif,
        duration,
        final_pause,
    )

    # Check if GIF exists
    assert os.path.exists(output_gif), "GIF was not created!"

    # Open the GIF and check properties
    with Image.open(output_gif) as gif:
        assert gif.format == "GIF", "Output file is not a GIF!"
        assert gif.n_frames == len(
            test_images
        ), "Incorrect number of frames in GIF!"

    print(f"Test passed for duration={duration}, final_pause={final_pause}")
