from inspect import getargs
import json
from pathlib import Path

import nrrd
import numpy as np
import SimpleITK as sitk
import tempfile
from functools import lru_cache

from toinstance.kernel_selection import get_np_arr_from_kernel, kernel_choices
from toinstance.utils import TAB20
from skimage import morphology as morph


def create_instance_map_of_semantic_map(semantic_array: np.ndarray, cc_kwargs: dict) -> dict[str, list[np.ndarray]]:
    f"""
    Transform a semantic map into an instance map through connected components analysis.
    :param semantic_array: Semantic map
    :param cc_kwargs: Keyword arguments for connected components analysis. (`dilation_kernel`: {getargs(kernel_choices)}, `label_connectivity`: [1, 2, 3], `dilation_kernel_radius`: int)
    
    :return: Dictionary of class-wise binary instances
    """
    dilation_kernel: np.ndarray | None
    label_connectivity: int
    dilation_kernel_radius = cc_kwargs.get("dilation_kernel_radius", 3)
    dilation_kernel = cc_kwargs.get("dilation_kernel", "ball")
    assert dilation_kernel in getargs(kernel_choices)
    label_connectivity = cc_kwargs.get("label_connectivity", 2)
    assert label_connectivity in [1, 2, 3], "Label connectivity must be 1, 2 or 3."

    dilation_kernel = get_np_arr_from_kernel(dilation_kernel, radius=dilation_kernel_radius, dtype=np.uint8)

    class_wise_instances: dict[str, list[np.ndarray]] = {}
    for unique_class_id in np.unique(semantic_array):
        if unique_class_id == 0:
            continue  # Ignore background
        class_wise_bin_maps = []
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
        for i in range(1, max_id + 1):
            class_wise_bin_maps.append((label_ce == i).astype(np.uint16))
        class_wise_instances[unique_class_id] = class_wise_bin_maps
    return class_wise_instances


class InstanceNrrd:
    def __init__(self, array: np.ndarray, header: dict):
        """
        InstanceNrrd object to handle instance maps in nrrd format.
        The header must contain the appropriate key ("org.mitk.multilabel.segmentation.labelgroups") for the instance maps.
        If not provided correctly this init will raise an AssertionError.
        However easier inits are provided to create an InstanceNrrd object from a semantic map or a semantic image or a dictionary of instance maps.
        Recommended init is through `InstanceNrrd.from_semantic_img` or `InstanceNrrd.from_semantic_map` or `InstanceNrrd.from_binary_instance_maps`.

        :param array: np.ndarray of shape (n_instances, x, y, z)
        :param header: Nrrd header
        """
        self.array: np.ndarray = array
        self.header: dict = header
        self.array.setflags(write=False)
        # Check that the header contains expected keys.
        self._verify_meta_data()

    def to_file(self, filepath: str | Path):
        """
        Write the instance nrrd to disk.
        Recommended to save as `.in.nrrd` file to indicate instance it being instance nrrd.\
        :param filepath: Path to save the instance nrrd.
        """
        nrrd.write(str(filepath), self.array, self.header)

    def _read_img(self, path: str | Path) -> tuple[np.ndarray, dict]:
        """
        Reads the image from the path and returns the array and header.
        Uses SimpleITK to read images that are not nrrd and extracts the files as if they were nrrd.

        :param path: Path to the image.
        :return: Tuple of array and header.
        """
        if str(path).endswith(".nrrd"):
            return nrrd.read(path)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "temp.nrrd"
                sitk_image = sitk.ReadImage(path)
                sitk.WriteImage(sitk_image, str(temp_path))
                return nrrd.read(temp_path)

    def _verify_meta_data(self):
        """
        Check if the header contains the expected keys.
        Currently expects there to be a key "org.mitk.multilabel.segmentation.labelgroups" in the header.
        Moreover
        """
        assert (
            "org.mitk.multilabel.segmentation.labelgroups" in self.header
        ), "org.mitk.multilabel.segmentation.labelgroups"
        assert (
            len(self.header["org.mitk.multilabel.segmentation.labelgroups"]) == self.array.shape[0]
        ), "Number of groups in header and array do not match."

    def semantic_classes(self) -> set[str]:
        """Return all semantic classes in the nrrd file."""
        all_class_names = set()
        for groups in self.header["org.mitk.multilabel.segmentation.labelgroups"]:
            for label in groups["labels"]:
                all_class_names.add(label["name"])

        return all_class_names

    def get_instance_values_of_semantic_class(self, class_name: str) -> set[int]:
        """Return all instance values of a semantic class"""
        instance_values = set()
        for groups in self.header["org.mitk.multilabel.segmentation.labelgroups"]:
            for label in groups["labels"]:
                if label["name"] == class_name:
                    instance_values.add(label["value"])
        return instance_values

    def get_semantic_map(self, class_name: str) -> np.ndarray:
        """Return the semantic map of a specific class as binary map."""
        instance_values = self.get_instance_values_of_semantic_class(class_name)
        semantic_map = np.sum(np.isin(self.array, instance_values), axis=0)
        return semantic_map

    @lru_cache
    def get_instance_maps(self, class_name: str) -> list[np.ndarray]:
        """Return all instance maps of a specific class."""
        instance_values = self.get_instance_values_of_semantic_class(class_name)
        instance_maps = [np.sum(self.array == iv, axis=0) for iv in instance_values]
        return instance_maps

    def get_semantic_instance_maps(self) -> dict[str, list[np.ndarray]]:
        """Return all semantic classes with their instance maps."""
        semantic_instance_maps = {}
        for class_name in self.semantic_classes():
            semantic_instance_maps[class_name] = self.get_instance_maps(class_name)
        return semantic_instance_maps

    def _update_array(self, instance_dict: dict[str, list[np.ndarray]] = None):
        """Updates the array with the new instance values."""

        self.array.setflags(write=True)
        self.array, self.header = self._arr_header_update_from_binmaps(instance_dict, self.header)
        self.array.setflags(write=False)

    def add_instance_map(self, instance_map: np.ndarray, class_name: str):
        """Add an instance map to the nrrd file."""
        instance_dict = self.get_semantic_instance_maps()
        instance_dict[class_name].append(instance_map)
        self._update_array(instance_dict)

    @staticmethod
    def _arr_header_update_from_binmaps(
        classwise_bin_maps: dict[str, list[np.ndarray]], header: dict
    ) -> tuple[np.ndarray, dict]:
        """Create an np.ndarray and an mitk compatible header from the instance maps."""
        final_arr = []
        lesion_header = []
        cnt = 1
        for class_name, instance_maps in classwise_bin_maps.items():
            for instance_map in instance_maps:
                lesion_header.append(
                    {
                        "labels": [
                            {
                                "color": {"type": "ColorProperty", "value": TAB20[(cnt - 1) % 20]},
                                "locked": True,
                                "name": f"{class_name}",
                                "opacity": 0.6,
                                "value": cnt,
                                "visible": True,
                            }
                        ]
                    }
                )
                final_arr.append(instance_map * cnt)
                cnt += 1
        final_arr = np.stack(final_arr, axis=0)
        header["org.mitk.multilabel.segmentation.labelgroups"] = json.dumps(lesion_header)
        return final_arr, header

    @staticmethod
    def from_binary_instance_maps(instance_dict: dict[str, list[np.ndarray]], header: dict) -> "InstanceNrrd":
        """
        Creates an InstanceNrrd object from a dictionary from a dictionary holding all instances for each semantic class.
        """
        final_arr, header = InstanceNrrd._arr_header_update_from_binmaps(instance_dict, header)
        return InstanceNrrd(final_arr, header)

    @staticmethod
    def from_semantic_map(
        semantic_map: np.ndarray, header: dict, do_cc: bool = False, cc_kwargs: dict = None
    ) -> "InstanceNrrd":
        """
        Creates an InstanceNrrd object from a semantic map.

        :param semantic_map: Semantic map
        :param header: Nrrd header
        :param do_cc: Whether to perform connected components analysis -- Otherwise whole semantic map is considered one instance.
        :param cc_kwargs: Keyword arguments for connected components analysis. (`dilation_kernel`: {getargs(kernel_choices)}, `label_connectivity`: [1, 2, 3], `dilation_kernel_radius`: int)
        """
        instance_dict: dict[str, list[np.ndarray]]
        if do_cc:
            instance_dict = create_instance_map_of_semantic_map(semantic_map, cc_kwargs)
        else:
            instance_dict = {}
            for class_name in np.unique(semantic_map):
                instance_dict[str(class_name)] = [semantic_map == class_name]
        return InstanceNrrd.from_binary_instance_maps(instance_dict, header)

    @staticmethod
    def from_semantic_img(
        semantic_img_path: str | Path, do_cc: bool = False, cc_kwargs: dict = None
    ) -> "InstanceNrrd":
        """
        Creates an InstanceNrrd object from a semantic image.

        :param semantic_img_path: Path to the semantic image.
        :param do_cc: Whether to perform connected components analysis -- Otherwise whole semantic map is considered one instance.
        :param cc_kwargs: Keyword arguments for connected components analysis. (`dilation_kernel`: {getargs(kernel_choices)}, `label_connectivity`: [1, 2, 3], `dilation_kernel_radius`: int)
        """
        semantic_array, header = InstanceNrrd._read_img(semantic_img_path)
        if do_cc:
            instance_dict = create_instance_map_of_semantic_map(semantic_array, cc_kwargs)
            return InstanceNrrd.from_binary_instance_maps(instance_dict, header)
        else:
            return InstanceNrrd.from_semantic_map(semantic_array, header)

    @staticmethod
    def from_innrrd(filepath: str | Path) -> "InstanceNrrd":
        """
        Read a native `in.nrrd` file from disk and return an `InstanceNrrd` object.

        :param filepath: Path to the instance `.in.nrrd`.
        """
        array, header = nrrd.read(filepath)
        return InstanceNrrd(array, header)
