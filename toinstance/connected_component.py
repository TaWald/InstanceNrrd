from copy import deepcopy
from functools import partial
import shutil
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import os
from skimage import morphology as morph
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from toinstance.naming_conventions import INSTANCE_SEG_PATTERN, SEMANTIC_SEG_PATTERN


def run_connected_components(
    samples: list[Path],
    output_dir: str,
    dilation_kernel: np.ndarray | None,
    label_connectivity: int,
    n_processes: int = 1,
    overwrite: bool = False,
) -> list[tuple[Path, Path]]:
    """Does connected component analysis on either the resampled or the raw data."""
    print("Creating Groundtruths intances.")
    partial_sem2ins = partial(
        create_instances_of_semantic_map,
        output_path=output_dir,
        dilation_kernel=dilation_kernel,
        label_connectivity=label_connectivity,
        overwrite=overwrite,  # Maybe unnecessary?
    )
    if n_processes == 1:
        converted_files = [partial_sem2ins(sample) for sample in tqdm(samples)]
    else:
        converted_files = process_map(partial_sem2ins, samples, max_workers=n_processes)
    return converted_files  # Return the paths to the created instances


def create_instances_of_semantic_map(
    semantic_seg_path: str | Path,
    output_path: str | Path,
    dilation_kernel: np.ndarray | None,
    label_connectivity: int,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Creates the label of the groundtruth by doing a connected components analysis (only on the CE).

    :param training_data: Path to sample sample
    :param output_path: Current path to save the Labeled Groundtruth to
    :param dilation_kernel: Structure to dilate Contrast enhanced pixels by
    :param label_kernel: Connectivity of labeling
    :param dilation_size: Indicator on how big the
    :param segmentation_id_to_merge: Segmentation id used to create the instances
    :return:
    """

    output_path = Path(output_path)
    # ------------------------- Loading Prediction Sample ------------------------ #
    semantic_seg_im = sitk.ReadImage(semantic_seg_path)
    semantic_array = deepcopy(sitk.GetArrayFromImage(semantic_seg_im))

    # --------------------------- Prepare Writing Paths -------------------------- #
    sample_name = os.path.basename(semantic_seg_path)
    case_id = sample_name.split(".")[0]
    instance_name = case_id + INSTANCE_SEG_PATTERN + ".nii.gz"
    semantic_name = case_id + SEMANTIC_SEG_PATTERN + ".nii.gz"
    instance_output_path = output_path / instance_name
    semantic_output_path = output_path / semantic_name

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if instance_output_path.exists() and semantic_output_path.exists() and not overwrite:
        return

    # ------------------------ Start creation of instances ----------------------- #
    unique_classes = np.unique(semantic_array)
    instance_arr = np.zeros_like(semantic_array).astype(np.int32)
    max_inst_id = 0
    for unique_class_id in unique_classes:
        if unique_class_id == 0:
            continue  # Ignore background
        bin_ce_res_pd_np = semantic_array == unique_class_id
        if dilation_kernel is not None:
            ce_closed_pd = morph.binary_dilation(image=bin_ce_res_pd_np, footprint=dilation_kernel).astype(np.int32)
        else:
            ce_closed_pd = bin_ce_res_pd_np
        # label_ce, _ = nd.label(input=ce_closed_pd, structure=label_kernel)
        label_ce, max_id = morph.label(
            label_image=ce_closed_pd,
            background=0,
            return_num=True,
            connectivity=label_connectivity,
        )
        instance_id_label = (bin_ce_res_pd_np * (label_ce + max_inst_id)).astype(np.int32)
        instance_arr += instance_id_label
        max_inst_id += max_id

    # ---------------------------------- Writing --------------------------------- #
    output_prediction_1 = sitk.GetImageFromArray(instance_arr)
    output_prediction_1.SetOrigin(semantic_seg_im.GetOrigin())
    output_prediction_1.SetDirection(semantic_seg_im.GetDirection())
    output_prediction_1.SetSpacing(semantic_seg_im.GetSpacing())

    sitk.WriteImage(output_prediction_1, instance_output_path)
    shutil.copy(semantic_seg_path, semantic_output_path)
    return (semantic_output_path, instance_output_path)
