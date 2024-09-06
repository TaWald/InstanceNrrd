import json
from pathlib import Path
from toinstance.naming_conventions import extensions
import numpy as np

TAB20 = [
    [0.12156862745098039, 0.4666666666666667, 0.7058823529411765],
    [0.6823529411764706, 0.7803921568627451, 0.9098039215686274],
    [1.0, 0.4980392156862745, 0.054901960784313725],
    [1.0, 0.7333333333333333, 0.47058823529411764],
    [0.17254901960784313, 0.6274509803921569, 0.17254901960784313],
    [0.596078431372549, 0.8745098039215686, 0.5411764705882353],
    [0.8392156862745098, 0.15294117647058825, 0.1568627450980392],
    [1.0, 0.596078431372549, 0.5882352941176471],
    [0.5803921568627451, 0.403921568627451, 0.7411764705882353],
    [0.7725490196078432, 0.6901960784313725, 0.8352941176470589],
    [0.5490196078431373, 0.33725490196078434, 0.29411764705882354],
    [0.7686274509803922, 0.611764705882353, 0.5803921568627451],
    [0.8901960784313725, 0.4666666666666667, 0.7607843137254902],
    [0.9686274509803922, 0.7137254901960784, 0.8235294117647058],
    [0.4980392156862745, 0.4980392156862745, 0.4980392156862745],
    [0.7803921568627451, 0.7803921568627451, 0.7803921568627451],
    [0.7372549019607844, 0.7411764705882353, 0.13333333333333333],
    [0.8588235294117647, 0.8588235294117647, 0.5529411764705883],
    [0.09019607843137255, 0.7450980392156863, 0.8117647058823529],
    [0.6196078431372549, 0.8549019607843137, 0.8980392156862745],
]


def create_mitk_nrrd(nrrd_arrs: dict[str, np.ndarray], nrrd_header: dict) -> tuple[np.ndarray, dict]:
    """Add MITK header information to the nrrd header.
    Allows visualizing the nrrd image nicely in MITK."""
    # ------------------------- Edit basic headers ------------------------- #
    nrrd_header["dimension"] += 1
    nrrd_header["type"] = "unsigned short"
    nrrd_header["sizes"] = [len(nrrd_arr)] + nrrd_header["sizes"]
    nrrd_header["space directions"] = [None] + nrrd_header["space directions"].tolist()
    nrrd_header["kinds"] = ["vector"] + nrrd_header["kinds"]
    nrrd_header["encoding"] = "gzip"
    # --------------------------- MITK Specific Headers -------------------------- #
    nrrd_header["modality"] = "org.mitk.multilabel.segmentation"

    lesion_header = []
    cnt = 1
    # There may be overlap or no overlap between the lesions.
    #   If there is overlap, another channel will be created.
    for class_name, nrrd_arr in nrrd_arrs.items():
        lesion_header.append(
            {
                "labels": [
                    {
                        "color": {"type": "ColorProperty", "value": TAB20[(cnt - 1) % 20]},
                        "locked": True,
                        "name": f"Class {class_name}",
                        "opacity": 0.6,
                        "value": cnt,
                        "visible": True,
                    }
                ]
            }
        )
        cnt += 1
    nrrd_header["org.mitk.multilabel.segmentation.labelgroups"] = json.dumps(lesion_header)
    nrrd_header["org.mitk.multilabel.segmentation.unlabeledlabellock"] = 0
    nrrd_header["org.mitk.multilabel.segmentation.version"] = 1
    mitk_compatible_arr = np.stack(nrrd_arrs, axis=0)
    return mitk_compatible_arr, nrrd_header


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
