# InstanceNrrd / ToInstance
WIP: This is a small repo that contains two functionalities:

1. `InstanceNrrd` - This is an image format that allows saving overlapping segmentation maps of 3D images in a format that is viewable using the [Medical Imaging ToolKit (MITK)](https://www.mitk.org/wiki/The_Medical_Imaging_Interaction_Toolkit_(MITK))  
2. Additionally it allows converting 3D semantic segmentation maps into a instance segmentation map using connected components analysis.

Currently the name of the repo and the package is `toinstance` but the main class is called `InstanceNrrd`. Naming will be changed soon.

---

## Installation

Clone this repository and install it locally:
```bash
pip install toinstance
```

```bash
git clone https://github.com/TaWald/InstanceNrrd.git
cd InstanceNrrd
pip install -e .
```
then you can use it via

## Usage

There are two ways this repository can be used.
The first is by creating using the `InstanceNrrd` class directly in python, which allows you to create, modify and save instance maps.

### InstanceNrrd

```python
from toinstance import InstanceNrrd

nrrd_ar, nrrd_header = nrrd.read(path_to_any_3d_image)
# Load and convert a semantic map to instances through a connected component analysis. 
innrrd = InstanceNrrd.from_semantic_map(nrrd_ar, nrrd_header, do_cc=True)
# Add some other instance to it - can be overlapping
innrrd.add_instance_map(some_bin_array, some_class_id)
# Save the (pot. overalpping) instance maps to a file.
innrrd.to_file("/some/target_path/instance.in.nrrd")
```

### CLI
You can also create instance maps from semantic segmentation maps directly from the CLI by calling the `toinstance` command:

```bash
# Convert a file or a dir to instance segmentations
toinstance -i <path_to_file_or_dir_with_files> -o <path_to_output_dir>
```

Moreover you can specify the following arguments:
 - `"-lc", "--label_connectivity"` - The connectivity for the connected components can be `[1, 2, 3]` --  1: only face 2: face and edge 3: face, edge, and corner 
 - `"-dk", "--dilation_kernel"` - Can be used pre-labeling to dilate the semantic map and allow connecting close-by points even without corner neighbourhood. Can be `["ball", "cube", "diamond", "octahedron"]`
 - `"-dkr", "--dilation_kernel_radius"` - The radius of the dilation kernel. Larger Kernels merges instances that are further away -- (Integer) Voxel radius of the kernel
 - `"-p", "--processes"` - The number of processes to use for multiprocessing -- (Integer) specifying the number of processes to use
 - `"-f", "--force_overwrite` - Whether to force overwrite existing files -- (Boolean) specifying whether to force overwrite existing files

This info can also be found by calling `toinstance --help`.
