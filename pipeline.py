import batch_run_whole
import filter_faces
import create_csv

PATH_VIDEO = "/Users/Timon/Documents/Houston/video_features/extracting_FAUs/outpath/GH010383.MOV"
PATH_VIDEO = "/Users/Timon/Downloads/TRBD001_20250603_134858.24253452.mp4"
PATH_VIDEO = "/Users/Timon/Library/CloudStorage/Box-Box/TRBD-Jamail/TRBD001/2025-07-17/GH012705_interview.MP4"
PATH_VIDEO = "/Users/Timon/Library/CloudStorage/Box-Box/TRBD-Jamail/TRBD001/2025-07-17/GH022705.MP4"
PATH_VIDEO = "/Users/Timon/Library/CloudStorage/Box-Box/TRBD-Jamail/TRBD001/2025-08-12/GH012774_interview.MP4"
batch_run_whole.main(PATH_VIDEO)
filter_faces.main(PATH_VIDEO)
create_csv.main(PATH_VIDEO)