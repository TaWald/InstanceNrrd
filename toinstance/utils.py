from pathlib import Path
from toinstance.naming_conventions import extensions


def get_readable_images_from_dir(dir_path: Path) -> list[Path]:
    """Returns a list of all readable images in a directory.

    :param dir_path: Path to directory
    :return: List of images
    """
    all_files = []
    for content in dir_path.iterdir():
        if content.is_file() and content.name.endswith(extensions):
            all_files.append(content)
    return all_files
