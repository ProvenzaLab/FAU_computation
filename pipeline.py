import batch_run_whole
import filter_faces
import create_csv
import os
from joblib import Parallel, delayed

# PATH_VIDEO = "/Users/Timon/Documents/Houston/video_features/extracting_FAUs/outpath/GH010383.MOV"
# PATH_VIDEO = "/Users/Timon/Downloads/TRBD001_20250603_134858.24253452.mp4"
# PATH_VIDEO = "/Users/Timon/Library/CloudStorage/Box-Box/TRBD-Jamail/TRBD001/2025-07-17/GH012705_interview.MP4"
# PATH_VIDEO = "/Users/Timon/Library/CloudStorage/Box-Box/TRBD-Jamail/TRBD001/2025-07-17/GH022705.MP4"
# PATH_VIDEO = "/Users/Timon/Library/CloudStorage/Box-Box/TRBD-Jamail/TRBD001/2025-08-12/GH012774_interview.MP4"

videos = os.listdir("/Users/Timon/Downloads/vids_process")

def run_pipeline_for_video(video_path):
    path_ = os.path.join("/Users/Timon/Downloads/vids_process", video_path)
    # check if path exists
    out_path = os.path.join("outdir", os.path.basename(path_)[:-4])
    path_npy = os.path.join(out_path, "au_predictions.npy")
    if os.path.exists(path_npy) is False:
        batch_run_whole.main(path_)
    path_faces = os.path.join(out_path, "filtered_results.csv")
    if os.path.exists(path_faces) is False:
        filter_faces.main(path_)
    path_csv = os.path.join(out_path, "full_au_results.csv")
    if os.path.exists(path_csv) is False:
        create_csv.main(path_)


# Run the pipeline for each video in parallel
Parallel(n_jobs=-1)(delayed(run_pipeline_for_video)(video) for video in videos)

# for PATH_VIDEO in videos[1:]:
#     run_pipeline_for_video(PATH_VIDEO)