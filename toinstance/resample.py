from multiprocessing import Pool
import os

import SimpleITK as sitk
import numpy as np

def resample(
    sample_name: str,
    prediction_location: str,
    groundtruth_location: str,
    resample_gt_location: str,
    resample_pd_location: str,
) -> None:
    """Takes all samples,
    assures that the prediction spacing and sampling is identical to the GT!

    Currently both are mapped to [1.0, 1.0, 1.0] is that intentional?
    Thought one usually calculates in groundtruth spacing.

    :param sample_name: Paths to the Tuple of Training modalities, Groundtruth and Predictions.
    :return:
    """

    gt_source_path = os.path.join(groundtruth_location, sample_name)
    pd_source_path = os.path.join(prediction_location, sample_name)
    if not os.path.exists(gt_source_path):
        raise FileNotFoundError("Could not find {}".format(gt_source_path))
    if not os.path.exists(pd_source_path):
        raise FileNotFoundError("Could not find {}".format(pd_source_path))

    gt_dir_path = conf.resample_gt_location
    pd_dir_path = conf.resample_pd_location

    os.makedirs(gt_dir_path, exist_ok=True)
    os.makedirs(pd_dir_path, exist_ok=True)

    gt_target_path = os.path.join(resample_gt_location, sample_name)
    pd_target_path = os.path.join(resample_pd_location, sample_name)

    if os.path.isfile(gt_target_path) and os.path.isfile(pd_target_path):
        return
    else:
        print("Resampled data not found.")
        gt: sitk.Image = sitk.ReadImage(gt_source_path)
        pred: sitk.Image = sitk.ReadImage(pd_source_path)

        gt_data = np.expand_dims(sitk.GetArrayFromImage(gt), 0)
        pred_data = np.expand_dims(sitk.GetArrayFromImage(pred), 0)

        sx_gt, sy_gt, sz_gt = gt.GetSpacing()[::-1]
        sx_pd, sy_pd, sz_pd = pred.GetSpacing()[::-1]

        if gt.GetSpacing() != pred.GetSpacing():
            print("GT and PD have different spacings.")

        target_spacing = [1.0, 1.0, 1.0]

        ######### Resampling  #########
        #   In principle the groundtruth should not be resampled, as
        #   This is the groundtruth that we care about.

        _, resampled_gt = resample_patient(
            data=None,
            seg=gt_data,
            original_spacing=(sx_gt, sy_gt, sz_gt),
            target_spacing=target_spacing,
        )
        _, resampled_pd = resample_patient(
            data=None,
            seg=pred_data,
            original_spacing=(sx_pd, sy_pd, sz_pd),
            target_spacing=target_spacing,
        )

        gt_img_res_itk = sitk.GetImageFromArray(np.squeeze(resampled_gt, axis=0))
        pd_img_res_itk = sitk.GetImageFromArray(np.squeeze(resampled_pd, axis=0))

        gt_img_res_itk.SetOrigin(gt.GetOrigin())
        gt_img_res_itk.SetDirection(gt.GetDirection())
        gt_img_res_itk.SetSpacing(target_spacing[::-1])

        pd_img_res_itk.SetOrigin(pred.GetOrigin())
        pd_img_res_itk.SetDirection(pred.GetDirection())
        pd_img_res_itk.SetSpacing(target_spacing[::-1])

        sitk.WriteImage(gt_img_res_itk, gt_target_path)
        sitk.WriteImage(pd_img_res_itk, pd_target_path)

        ## Safe the resampling steps to visualize them ##
        ## Maybe even safe the morphological operation segmentations
        return


def run_resample(
    data_pair: conf.DataPair,
    sorted_samples: list,
    n_processes: int = 1,
):
    """Loads the original Data from the data location and resamples them.
    After resampling saves them to the resample location

    :return:
    """
    print("Starting resampling")
    if n_processes == 1:
        for sample in sorted_samples:
            resample(
                sample_name=sample,
                groundtruth_location=data_pair.gt_p,
                prediction_location=data_pair.pd_p,
                resample_gt_location=data_pair.resample_p_gt,
                resample_pd_location=data_pair.resample_p_pd,
            )
    else:
        with Pool(n_processes) as p:
            p.starmap(
                resample,
                zip(
                    list(sorted_samples),
                    [data_pair.pd_p for _ in sorted_samples],
                    [data_pair.gt_p for _ in sorted_samples],
                    [data_pair.resample_p_gt for _ in sorted_samples],
                    [data_pair.resample_p_pd for _ in sorted_samples],
                ),
            )


if __name__ == "__main__":
    print("Wrong file")
