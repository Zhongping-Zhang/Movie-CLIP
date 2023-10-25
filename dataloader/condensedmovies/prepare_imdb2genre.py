import argparse
import pandas as pd
from os.path import join, basename, dirname

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_dir',type=str,default='data/CondensedMovies/metadata',help="directory to metadata files of Condensed Movies")
parser.add_argument('--output_dir', type=str, default='data/CondensedMovies/MovieCLIP_features', help='directory to save processed features for MovieCLIP')
parser.add_argument('--include_keyframe_path', type=bool, default=False)
parser.add_argument('--debug', type=bool, default=False)
args = parser.parse_args()


movie_info = pd.read_csv(join(args.metadata_dir,"movie_info.csv"))
clip_info = pd.read_csv(join(args.metadata_dir, 'clips.csv'))

df = clip_info.merge(movie_info, on="imdbid")#.merge(frame, on='videoid')

if args.include_keyframe_path:
    from glob import glob
    frame_file_folder = join(dirname(args.metadata_dir),"MovieCLIP_features/keyFrames")
    frame_files = glob(frame_file_folder + '/*.csv')
    frame_dfs = [pd.read_csv(file) for file in frame_files]
    frame = pd.concat(frame_dfs, ignore_index=True, sort=False)
    df = df.merge(frame, on='videoid')
    df['frame_path'] = df['frame_path'].str.replace('/research/zpzhang/DATA/CondensedMovies',args.output_dir)
    cols = ['imdbid', 'frame_path', 'genre']
    df[cols].to_csv(join(args.output_dir,"imdbid2genre_frame.txt"), index=False, header=False, sep='\t')

else:
    cols=['imdbid','genre']
    df[cols].to_csv(join(args.output_dir,"imdbid2genre.txt"),index=False,header=False,sep='\t')

print("complete")














