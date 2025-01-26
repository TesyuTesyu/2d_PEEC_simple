import subprocess

#ffmpegで処理（インストール&pathを通してから動く）.
input_path=r"???\PEEC_work_space\fig"#連番画像が入っているフォルダのパス.
out_path=input_path+"\\out.mp4"

command = f'ffmpeg -framerate 5 -i "{input_path+"\\"+"%03d_current.png"}" -vcodec libx264 -pix_fmt yuv420p -r 15 {out_path}'

subprocess.call(command, shell=True)
