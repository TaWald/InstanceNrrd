from functools import partial


from pathlib import Path

import os

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from toinstance.instance_nrrd import InstanceNrrd
from toinstance.kernel_selection import kernel_choices


def run_connected_components(
    samples: list[Path],
    output_dir: str,
    dilation_kernel: kernel_choices,
    dilation_radius: int,
    label_connectivity: int,
    n_processes: int = 1,
    overwrite: bool = False,
) -> list[Path]:
    """Does connected component analysis on either the resampled or the raw data."""
    print("Creating Groundtruths intances.")
    cc_kwargs = {
        "label_connectivity": label_connectivity,
        "dilation_kernel": dilation_kernel,
        "dilation_kernel_radius": dilation_radius,
    }
    partial_sem2ins = partial(
        create_instances_of_semantic_map_path,
        output_path=output_dir,
        cc_kwargs=cc_kwargs,
        overwrite=overwrite,  # Maybe unnecessary?
    )
    if n_processes == 1:
        converted_files = [partial_sem2ins(sample) for sample in tqdm(samples)]
    else:
        converted_files = process_map(partial_sem2ins, samples, max_workers=n_processes)
    return converted_files  # Return the paths to the created instances


def create_instances_of_semantic_map_path(
    semantic_seg_path: str | Path,
    output_path: str | Path,
    cc_kwargs: dict,
    overwrite: bool = False,
) -> Path:
    """Creates the label of the groundtruth by doing a connected components analysis (only on the CE).

    :param training_data: Path to sample sample
    :param output_path: Current path to save the Labeled Groundtruth to
    :param dilation_kernel: Structure to dilate Contrast enhanced pixels by
    :param label_kernel: Connectivity of labeling
    :param dilation_size: Indicator on how big the
    :param segmentation_id_to_merge: Segmentation id used to create the instances
    :return:
    """
    sample_name = os.path.basename(semantic_seg_path)
    case_id = sample_name.split(".")[0]
    instance_name = case_id + ".in.nrrd"

    output_path = Path(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    instance_output_path = output_path / instance_name
    # ------------------------- Loading Prediction Sample ------------------------ #
    innrrd = InstanceNrrd.from_semantic_img(semantic_seg_path, do_cc=True, cc_kwargs=cc_kwargs)
    innrrd.to_file(instance_output_path)

    return instance_output_path
