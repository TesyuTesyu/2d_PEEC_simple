import numpy as np
import matplotlib.pyplot as plt
import csv

csv_folder_path=r"???\PEEC_work_space"

with open(csv_folder_path+"\\freq.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    matf = np.array(list(reader)).astype(float)
with open(csv_folder_path+"\\im_Z.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    im_matZ = np.array(list(reader)).astype(float)
with open(csv_folder_path+"\\re_Z.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    re_matZ = np.array(list(reader)).astype(float)

im_matZ=np.array(im_matZ[0])
re_matZ=np.array(re_matZ[0])
matf=np.array(matf[0])

fig, ax = plt.subplots(nrows=2, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
#ax[0,0].set_title("")
ax[0,0].plot(matf,re_matZ,"k-")
ax[0,0].set_xscale('log')
#ax[0,0].set_yscale('log')
ax[0,0].set_ylabel("Re(Z) [Ohm]")
ax[1,0].plot(matf,im_matZ,"k-")
ax[1,0].set_xscale('log')
#ax[1,0].set_yscale('log')
ax[1,0].set_ylabel("Im(Z) [Ohm]")
ax[1,0].set_xlabel("Frequency [Hz]")

fig, ax = plt.subplots(nrows=2, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
#ax[0,0].set_title("")
ax[0,0].plot(matf,np.sqrt(re_matZ**2+im_matZ**2),"k-")
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
ax[0,0].set_ylabel("Magnitude [Ohm]")
ax[1,0].plot(matf,np.arctan2(im_matZ,re_matZ)*180/np.pi,"k-")
ax[1,0].set_xscale('log')
ax[1,0].set_ylabel("Phase [deg]")
ax[1,0].set_xlabel("Frequency [Hz]")
plt.show()