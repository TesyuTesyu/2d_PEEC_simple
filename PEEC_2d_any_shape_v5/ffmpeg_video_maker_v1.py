import subprocess

input_path=r"???\PEEC_work_space\fig"
out_path=input_path+"\\out_voltage.mp4"

command = f'ffmpeg -framerate 5 -i "{input_path+"\\"+"%03d_voltage.png"}" -vcodec libx264 -pix_fmt yuv420p -r 15 {out_path}'

subprocess.call(command, shell=True)
