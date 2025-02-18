import os
import urllib.request

from tqdm.auto import tqdm


def download_with_progress(url: str, output_path: str):
    """
    Download a file from the given URL with a progress bar.
    """
    # Open the URL and get the file size from the headers
    response = urllib.request.urlopen(url)
    total_size = int(response.info().get("Content-Length", -1))
    block_size = 1024  # 1 KB chunks

    # Create the progress bar
    with tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=os.path.basename(output_path),
    ) as pbar:
        with open(output_path, "wb") as out_file:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                pbar.update(len(buffer))
