# Simple Instance
This is a small repo to convert your 3D semantic instance segmentations into instance masks.
You pass a path to an image or dir containing images `<some_path>/<some_semantic_seg_file_id>.nii.gz` and
 it creates instances outputting each file into two new files:
- `<some_path>/<some_semantic_seg_file_id>_sem.nii.gz` - The same semantic segmentation map (What it got as input) 
- `<some_path>/<some_semantic_seg_file_id>_inst.nii.gz` - The instanced file that containing the instance labels.
By doing so it can be easily introspected side-by-side with the semantic map in a 3D image viewer.

## Installation
```bash
git clone
cd toinstance
pip install -e .
```

## Usage
### CLI
To use you from the CLI you can call `toinstance`:

```bash
# Convert a file or a dir to instance segmentations
toinstance -i <path_to_file_or_dir_with_files> -o <path_to_output_dir>
```

Moreover you can specify the following arguments:`
 - `"-lc", "--label_connectivity"` - The connectivity for the connected components can be `[1, 2, 3]` --  1: only face 2: face and edge 3: face, edge, and corner 
 - `"-dk", "--dilation_kernel"` - Can be used pre-labeling to dilate the semantic map and allow connecting close-by points even without corner neighbourhood. Can be `["ball", "cube", "diamond", "octahedron"]`
 - `"-dkr", "--dilation_kernel_radius"` - The radius of the dilation kernel. Larger Kernels merges instances that are further away -- (Integer) Voxel radius of the kernel
 - `"-p", "--processes"` - The number of processes to use for multiprocessing -- (Integer) specifying the number of processes to use
 - `"-f", "--force_overwrite` - Whether to force overwrite existing files -- (Boolean) specifying whether to force overwrite existing files

This info can also be found by calling `toinstance --help`.

### API
If you want you can also start the conversion from within python directly:

```python
from sem2ins.predict import create_instance
# Converts either a single or all files in a directory to instances.
output_paths = create_instance(input_path=..., output_path= ...)
# Output paths contains list[tuple[Path, Path]] where the first path is the semantic segmentation and the second the instance segmentation.
```

### Current Limitations
- Currently only supports `.nii.gz` / `.nrrd` files (But can be easily extended to support more)
- Currently only supports 3D semantic segmentation maps (But can be easily extended to support more)

This is a small repo that currently only serves to keep instance creation from semantic segmentation maps simple and easy to use. It is not meant to be a full fledged library (yet) and just keeps the instance generation outside of the `nneval` repository. 

If you want to extend the functionality of this, feel free to open a PR. 