import batch_run_whole
import filter_faces
import create_csv

# PATH_VIDEO = "/Users/Timon/Documents/Houston/video_features/extracting_FAUs/outpath/GH010383.MOV"
# PATH_VIDEO = "/Users/Timon/Downloads/TRBD001_20250603_134858.24253452.mp4"
# PATH_VIDEO = "/Users/Timon/Library/CloudStorage/Box-Box/TRBD-Jamail/TRBD001/2025-07-17/GH012705_interview.MP4"
# PATH_VIDEO = "/Users/Timon/Library/CloudStorage/Box-Box/TRBD-Jamail/TRBD001/2025-07-17/GH022705.MP4"
# PATH_VIDEO = "/Users/Timon/Library/CloudStorage/Box-Box/TRBD-Jamail/TRBD001/2025-08-12/GH012774_interview.MP4"

PATH_LIST = [
    "mnt/datalake/data/aDBS-49155/aDBS005 Recordings/2019-12-19/Camera 1/GH010295.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS007 Recordings/2020-11-11/Camera 1/GH010642.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS007 Recordings/2020-12-11/Camera 1/GH010671.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS008 Recordings/2023-07-26/Camera 1/GH012109.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS009 Recordings/2022-07-12/Camera 1/GH011478.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS010 Recordings/2022-06-23/Camera 1/GH011460.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS010 Recordings/2024-03-21/Camera 1/GH012364.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS011 Recordings/2023-01-10/Camera 1/GH011810.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS012 Recordings/2023-09-19/Camera 1/GH012161.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS012 Recordings/2023-09-26/Camera 1/GH012169.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS012 Recordings/2023-10-05/Camera 1/GH012178.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS012 Recordings/2023-11-15/Camera 1/GH012226.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS012 Recordings/2023-12-18/Camera 1/GH012239.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS012 Recordings/2024-01-29/Camera 1/GH012294.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS012 Recordings/2024-02-12/Camera 1/GH012307.MP4",
    "mnt/datalake/data/aDBS-49155/aDBS012 Recordings/2024-03-19/Camera 1/GH012354.MP4",
]

PATH_LIST_WINDOWS = [
    r"Z:\aDBS-49155\aDBS005 Recordings\2019-12-19\Camera 1\GH010295.MP4",
    r"Z:\aDBS-49155\aDBS007 Recordings\2020-11-11\Camera 1\GH010642.MP4",
    r"Z:\aDBS-49155\aDBS007 Recordings\2020-12-11\Camera 1\GH010671.MP4",
    r"Z:\aDBS-49155\aDBS008 Recordings\2023-07-26\Camera 1\GH012109.MP4",
    r"Z:\aDBS-49155\aDBS009 Recordings\2022-07-12\Camera 1\GH011478.MP4",
    r"Z:\aDBS-49155\aDBS010 Recordings\2022-06-23\Camera 1\GH011460.MP4",
    r"Z:\aDBS-49155\aDBS010 Recordings\2024-03-21\Camera 1\GH012364.MP4",
    r"Z:\aDBS-49155\aDBS011 Recordings\2023-01-10\Camera 1\GH011810.MP4",
    r"Z:\aDBS-49155\aDBS012 Recordings\2023-09-19\Camera 1\GH012161.MP4",
    r"Z:\aDBS-49155\aDBS012 Recordings\2023-09-26\Camera 1\GH012169.MP4",
    r"Z:\aDBS-49155\aDBS012 Recordings\2023-10-05\Camera 1\GH012178.MP4",
    r"Z:\aDBS-49155\aDBS012 Recordings\2023-11-15\Camera 1\GH012226.MP4",
    r"Z:\aDBS-49155\aDBS012 Recordings\2023-12-18\Camera 1\GH012239.MP4",
    r"Z:\aDBS-49155\aDBS012 Recordings\2024-01-29\Camera 1\GH012294.MP4",
    r"Z:\aDBS-49155\aDBS012 Recordings\2024-02-12\Camera 1\GH012307.MP4",
    r"Z:\aDBS-49155\aDBS012 Recordings\2024-03-19\Camera 1\GH012354.MP4",
]
for PATH_VIDEO in PATH_LIST:
    batch_run_whole.main(PATH_VIDEO)
    filter_faces.main(PATH_VIDEO)
    create_csv.main(PATH_VIDEO)