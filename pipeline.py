import batch_run_whole
import filter_faces
import create_csv

PATH_VIDEO = "/Users/Timon/Documents/Houston/video_features/extracting_FAUs/outpath/GH010383.MP4"
batch_run_whole.main(PATH_VIDEO)
filter_faces.main(PATH_VIDEO)
create_csv.main(PATH_VIDEO)