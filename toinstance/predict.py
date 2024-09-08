from argparse import ArgumentParser
from pathlib import Path
from typing import get_args
from warnings import warn
from toinstance.connected_component import run_connected_components
from toinstance.kernel_selection import kernel_choices, get_kernel_from_str, get_np_arr_from_kernel

from toinstance.utils import get_readable_images_from_dir
import numpy as np


def create_instance(
    input_path: Path,
    output_dir: Path,
    label_connectivity: int = 2,
    dilation_kernel: kernel_choices | None = "ball",
    dilation_kernel_radius: int = 3,
    processes: int = 1,
    overwrite: bool = False,
) -> list[tuple[Path, Path]]:
    """
    Either convert a single segmentation or multiple segmentations in a directory into instances.
    Instances are written to disk in the output_dir. They are split into semantic segmentation and instances,
    for easy visibility in a viewer.
    """
    if input_path.is_dir():
        all_files = get_readable_images_from_dir(input_path)
    else:
        all_files = [input_path]
    if len(all_files) == 0:
        warn(f"No files found in {input_path}")
        return
    print(f"Found {len(all_files)} files to process.")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    dk = get_kernel_from_str(dilation_kernel)
    dilation_kernel = get_np_arr_from_kernel(dk, radius=dilation_kernel_radius, dtype=np.uint8)
    return run_connected_components(all_files, output_dir, dilation_kernel, label_connectivity, processes, overwrite)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        help="Path to file or dir containing the segmentation(s)",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        help="Path to the output directory",
    )
    parser.add_argument(
        "-lc",
        "--label_connectivity",
        type=int,
        choices=[1, 2, 3],
        default=2,
        required=False,
        help="Desired connectivity for labeling. 1: cross, 2: ball, 3: rectangle",
    )
    parser.add_argument(
        "-dk",
        "--dilation_kernel",
        type=str,
        choices=get_args(kernel_choices),
        default="ball",
        required=False,
        help="Dilation kernel",
    )
    parser.add_argument(
        "-dkr",
        "--dilation_kernel_radius",
        type=int,
        default=3,
        required=False,
        help="Dilation kernel radius",
    )
    parser.add_argument("-p", "--processes", type=int, default=1, help="Number of processes to use.")
    parser.add_argument(
        "-f", "--force_overwrite", type=int, default=1, help="Will re-write files even if they exist."
    )

    args = parser.parse_args()

    input_path: Path
    output_path: Path
    input_path = args.input_path
    output_path = args.output_path
    # Check if these are absolute or relative paths. If relative, make them absolute.
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    create_instance(
        input_path=input_path,
        output_dir=output_path,
        label_connectivity=args.label_connectivity,
        dilation_kernel=args.dilation_kernel,
        dilation_kernel_radius=args.dilation_kernel_radius,
        processes=args.processes,
        overwrite=args.force_overwrite,
    )


if __name__ == "__main__":
    main()
