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
 - `"-dk", "--dilation_kernel"` - Can be used per-labeling to dilate foreground and allow connecting close-by points even without corner neighbourhood. Can be `["ball", "cube", "diamond", "octahedron"]`
 - `"-dkr", "--dilation_kernel_radius"` - The radius of the dilation kernel -- (Integer) specifying the voxel radius of the kernel
 - `"-p", "--processes"` - The number of processes to use for multiprocessing -- (Integer) specifying the number of processes to use
 - `"-f", "--force_overwrite` - Whether to force overwrite existing files -- (Boolean) specifying whether to force overwrite existing files

This info can also be found by calling `toinstance --help`.

### API


```python
from sem2ins.predict import file2instance, dir2instance
file2instance -i <path_to_file>.nii.gz -o <path_to_output_dir>
# Convert a directory of files to instance segmentation
dir2instance -i <path_to_dir> -o <path_to_output_dir>
```

