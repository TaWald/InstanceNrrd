import json
from pathlib import Path
from typing import get_args

import nrrd
import numpy as np
import SimpleITK as sitk
import tempfile

from toinstance.kernel_selection import get_kernel_from_str, get_np_arr_from_kernel, kernel_choices
from toinstance.utils import TAB20
from skimage import morphology as morph


def create_instance_map_of_semantic_map(semantic_array: np.ndarray, cc_kwargs: dict) -> dict[int, list[np.ndarray]]:
    f"""
    Transform a semantic map into an instance map through connected components analysis.
    :param semantic_array: Semantic map
    :param cc_kwargs: Keyword arguments for connected components analysis. (`dilation_kernel`: {get_args(kernel_choices)}, `label_connectivity`: [1, 2, 3], `dilation_kernel_radius`: int)
    
    :return: Dictionary of class-wise binary instances
    """
    dilation_kernel: np.ndarray | None
    label_connectivity: int
    dilation_kernel_radius = cc_kwargs.get("dilation_kernel_radius", 3)
    dilation_kernel = cc_kwargs.get("dilation_kernel", "ball")
    assert dilation_kernel in get_args(
        kernel_choices
    ), f"Dilation kernel {dilation_kernel} must be one of the available kernel choices: {get_args(kernel_choices)}"
    label_connectivity = cc_kwargs.get("label_connectivity", 2)
    assert label_connectivity in [1, 2, 3], "Label connectivity must be 1, 2 or 3."

    dk = get_kernel_from_str(dilation_kernel)
    dilation_kernel = get_np_arr_from_kernel(dk, radius=dilation_kernel_radius, dtype=np.uint8)

    class_wise_instances: dict[int, list[np.ndarray]] = {}
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
        # Remove the dilation from the label map
        label_ce = bin_ce_res_pd_np * label_ce
        for i in range(1, max_id + 1):
            class_wise_bin_maps.append((label_ce == i).astype(np.uint16))
        class_wise_instances[int(unique_class_id)] = class_wise_bin_maps

    total_instances = sum([len(v) for v in class_wise_instances.values()])
    if total_instances == 0:
        # Create an empty instance map if no instances are found. Otherwise stuff breaks.
        class_wise_instances = {0: [np.zeros_like(semantic_array)]}
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

    @staticmethod
    def _serialize_header(header: dict) -> str:
        """
        MITK needs this serialized.
        """
        if len(header["org.mitk.multilabel.segmentation.labelgroups"]) == 0:
            header.pop("org.mitk.multilabel.segmentation.labelgroups")
        else:
            header["org.mitk.multilabel.segmentation.labelgroups"] = json.dumps(
                header["org.mitk.multilabel.segmentation.labelgroups"]
            )
        return header

    def to_file(self, filepath: str | Path):
        """
        Write the instance nrrd to disk.
        Recommended to save as `.in.nrrd` file to indicate instance it being instance nrrd.\
        :param filepath: Path to save the instance nrrd.
        """
        nrrd.write(str(filepath), self.array, self._serialize_header(self.header))

    @staticmethod
    def _read_img(path: str | Path) -> tuple[np.ndarray, dict]:
        """
        Reads the image from the path and returns the array and header.
        Uses SimpleITK to read images that are not nrrd and extracts the files as if they were nrrd.

        :param path: Path to the image.
        :return: Tuple of array and header.
        """
        if str(path).endswith(".nrrd"):
            arr, header = nrrd.read(path)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "temp.nrrd"
                sitk_image = sitk.ReadImage(path)
                sitk.WriteImage(sitk_image, str(temp_path))
                arr, header = nrrd.read(temp_path)
        header["org.mitk.multilabel.segmentation.labelgroups"] = json.loads(
            header.get("org.mitk.multilabel.segmentation.labelgroups", "[]")
        )
        return arr, header

    def _verify_meta_data(self):
        """
        Check if the header contains the expected keys.
        Currently expects there to be a key "org.mitk.multilabel.segmentation.labelgroups" in the header.
        Moreover
        """
        assert (
            "org.mitk.multilabel.segmentation.labelgroups" in self.header
        ), "org.mitk.multilabel.segmentation.labelgroups"
        if len(self.header["org.mitk.multilabel.segmentation.labelgroups"]) != 0:
            assert (
                len(self.header["org.mitk.multilabel.segmentation.labelgroups"]) == self.array.shape[0]
            ), "Number of groups in header and array do not match."

    def semantic_classes(self) -> set[int]:
        """Return all semantic classes in the nrrd file."""
        all_class_names = set()
        for groups in self.header["org.mitk.multilabel.segmentation.labelgroups"]:
            for label in groups["labels"]:
                all_class_names.add(int(label["name"]))

        return all_class_names

    def get_instance_values_of_semantic_class(self, class_id: int) -> set[int]:
        """Return all instance values of a semantic class"""
        instance_values = set()
        for groups in self.header["org.mitk.multilabel.segmentation.labelgroups"]:
            for label in groups["labels"]:
                if int(label["name"]) == class_id:
                    instance_values.add(label["value"])
        return instance_values

    def get_semantic_map(self, class_id: int) -> np.ndarray:
        """Return the semantic map of a specific class as binary map."""
        instance_values = self.get_instance_values_of_semantic_class(class_id)
        semantic_map = np.sum(np.isin(self.array, instance_values), axis=0)
        return semantic_map

    def get_instance_maps(self, class_id: int) -> list[np.ndarray]:
        """Return all instance maps of a specific class."""
        instance_values = self.get_instance_values_of_semantic_class(class_id)
        instance_maps = [np.sum(self.array == iv, axis=0) for iv in instance_values]
        return instance_maps

    def get_semantic_instance_maps(self) -> dict[int, list[np.ndarray]]:
        """Return all semantic classes with their instance maps."""
        semantic_instance_maps = {}
        for class_id in self.semantic_classes():
            semantic_instance_maps[class_id] = self.get_instance_maps(class_id)
        return semantic_instance_maps

    def _update_array(self, instance_dict: dict[int, list[np.ndarray]] = None):
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
        classwise_bin_maps: dict[int, list[np.ndarray]], header: dict
    ) -> tuple[np.ndarray, dict]:
        """Create an np.ndarray and an mitk compatible header from the instance maps."""
        final_arr = []
        lesion_header = []
        cnt = 1

        # Indicator that there is no foreground at all
        if 0 in classwise_bin_maps:
            # If there is no foreground, we just return an empty instance map
            final_arr = classwise_bin_maps[0][0]
            header["innrrd.empty"] = 1
        else:
            for class_id, instance_maps in classwise_bin_maps.items():
                for instance_map in instance_maps:
                    lesion_header.append(
                        {
                            "labels": [
                                {
                                    "color": {"type": "ColorProperty", "value": TAB20[(cnt - 1) % 20]},
                                    "locked": True,
                                    "name": f"{int(class_id)}",
                                    "opacity": 0.6,
                                    "value": cnt,
                                    "visible": True,
                                }
                            ]
                        }
                    )
                    final_arr.append(instance_map * cnt)
                    header["innrrd.empty"] = 0
                    cnt += 1
            final_arr = np.stack(final_arr, axis=0)
        final_arr = final_arr.astype(np.uint16)
        # ---------------- Set the header in accordance to MITK format --------------- #
        header["org.mitk.multilabel.segmentation.labelgroups"] = lesion_header
        header["type"] = "unsigned short"
        header["encoding"] = "gzip"
        header["modality"] = "org.mitk.multilabel.segmentation"
        header["sizes"] = list(final_arr.shape)
        header["dimension"] = final_arr.ndim
        header["org.mitk.multilabel.segmentation.unlabeledlabellock"] = 0
        header["org.mitk.multilabel.segmentation.version"] = 1

        # Check if the image header already is in.nrrd
        if not header.get("innrrd", False):
            if header["innrrd.empty"] == 0:
                space_dirs = header["space directions"]
                # Header not in in.nrrd format, so we need to pre-pend stuff to edit general header infos.
                if isinstance(space_dirs, np.ndarray):
                    space_dirs = space_dirs.tolist()
                if header["space directions"][0] is not None:
                    header["space directions"] = [None] + list(space_dirs)  # Make sure it's a list
                if header["kinds"][0] != "vector":
                    header["kinds"] = ["vector"] + header["kinds"]
            header["innrrd"] = True

        return final_arr, header

    @staticmethod
    def from_binary_instance_maps(instance_dict: dict[int, list[np.ndarray]], header: dict) -> "InstanceNrrd":
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
        :param do_cc: Whether to perform connected components analysis -- Otherwise each value is one instance..
        :param cc_kwargs: Keyword arguments for connected components analysis. (`dilation_kernel`: {get_args(kernel_choices)}, `label_connectivity`: [1, 2, 3], `dilation_kernel_radius`: int)
        """
        instance_dict: dict[int, list[np.ndarray]]
        if do_cc:
            instance_dict = create_instance_map_of_semantic_map(semantic_map, cc_kwargs)
        else:
            instance_dict = {}
            for class_id in np.unique(semantic_map):
                instance_dict[int(class_id)] = [semantic_map == class_id]
        return InstanceNrrd.from_binary_instance_maps(instance_dict, header)

    @staticmethod
    def from_instance_map(instance_map: np.ndarray, header: dict, class_name: str) -> "InstanceNrrd":
        """
        Creates an InstanceNrrd object from an instance map.

        :param instance_map: Instance map, where each value is an instance.
        :param header: Nrrd header
        :param class_name: Class name all instances belong to.
        """
        bin_maps = [(instance_map == i).astype(np.uint16) for i in np.unique(instance_map) if i != 0]
        return InstanceNrrd.from_binary_instance_maps({class_name: bin_maps}, header)

    @staticmethod
    def from_semantic_img(
        semantic_img_path: str | Path, do_cc: bool = False, cc_kwargs: dict = None
    ) -> "InstanceNrrd":
        """
        Creates an InstanceNrrd object from a semantic image.

        :param semantic_img_path: Path to the semantic image.
        :param do_cc: Whether to perform connected components analysis -- Otherwise whole semantic map is considered one instance.
        :param cc_kwargs: Keyword arguments for connected components analysis. (`dilation_kernel`: {get_args(kernel_choices)}, `label_connectivity`: [1, 2, 3], `dilation_kernel_radius`: int)
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
        header["org.mitk.multilabel.segmentation.labelgroups"] = json.loads(
            header.get("org.mitk.multilabel.segmentation.labelgroups", "[]")
        )
        return InstanceNrrd(array, header)

    def get_spacing(self) -> list[list[float]]:
        """
        Get the spacing of the instance map.
        """
        return self.header["space directions"][1:]

    def set_spacing(self, spacing: list[list[float]]):
        """
        Set the spacing of the instance map.
        """
        self.header["space directions"] = [None] + spacing

    def get_origin(self) -> list[float]:
        """
        Get the origin of the instance map.
        """
        return self.header["space origin"]

    def set_origin(self, origin: list[float]):
        """
        Set the origin of the instance map.
        """
        self.header["space origin"] = origin

    def get_size(self) -> list[int]:
        """
        Get the size of the instance map.
        """
        return self.header["sizes"][1:]
