import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
import time
import copy
from matplotlib.colors import LogNorm
import csv
import PEEC_2d_any_shape_v4_make_matrix_v1

#---------------------定数の設定はここから.--------------------

# 真空の透磁率、誘電率、導体の抵抗率.
mu=4*np.pi*1e-7
eps= 8.854e-12
rho_ohm=1.68e-8

len_x=100e-3 #横方向の長さ[m].縦は画像のアスペクトで決まる.
size_x=50 #横方向の分割数.縦は画像のアスペクトで決まる.

tickness=0.01e-3 #導体面の厚さ[m].

matf=np.logspace(4,7,50)#周波数の配列.自由に変更.はじめは少ない数でテストした方がダメージが少ない.
#matf=np.linspace(0,3e9,50)

#画像の設定
color_th=100 #白か黒かをこの閾値で判定.
file_path=r"???\PEEC_test3.png" #ファイルパス.

#csvの保存場所.
csv_folder_path=r"???\PEEC_work_space"#いろいろ保存されるのでフォルダを作るほうがいい.
config_csv_file_path=csv_folder_path+"\\config.csv"#設定が保存される.

#グラフの保存場所.
fig_folder_path=csv_folder_path+"\\fig"#これだとcsv_folder_pathの下の"fig"という名前のフォルダに保存される.あらかじめ作成する.

#電圧源の位置.
#例：.
i_voltage=[1,24] #x方向の位置(?番目).
j_voltage=[9,9] #y方向の位置(?番目).
voltage=[1,0] #それぞれの電圧.
measure_node=[[0]]#インピーダンスの測定に用いるノード（複数の座標で一枚ならば全ての座標を入れる）.

#左右に一列の電極を貼り、左を1V、右を0Vとしたとき.
gjkl=37#電極の行数.
i_voltage=np.concatenate([np.zeros(gjkl,dtype="int"),49*np.ones(gjkl,dtype="int")])
j_voltage=np.concatenate([np.linspace(0,gjkl-1,gjkl,dtype="int"),np.linspace(0,gjkl-1,gjkl,dtype="int")])
voltage=np.concatenate([np.ones(gjkl,dtype="int"),np.zeros(gjkl,dtype="int")])
measure_node=[np.linspace(0,gjkl-1,gjkl,dtype="int")]#インピーダンスの測定に用いるノード.

#---------------------定数の設定はここまで.--------------------

#変数をCSVに保存.
with open(config_csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['mu', mu])
    writer.writerow(['eps', eps])
    writer.writerow(['rho_ohm', rho_ohm])
    writer.writerow(['len_x', len_x])
    writer.writerow(['size_x', size_x])
    writer.writerow(['tickness', tickness])
    writer.writerow(['color_th', color_th])
    writer.writerow(['file_path', file_path])
    writer.writerow(['csv_file_path', csv_folder_path])

#写真から形状を生成.
im_gray=np.array(Image.open(file_path).convert('L'))
im_y_size=len(im_gray)
im_x_size=len(im_gray[0])

len_y=len_x*im_y_size/im_x_size
size_y=int(size_x*im_y_size/im_x_size)

mat_shape=np.zeros([size_y,size_x],dtype="int")
for i in range(size_x):
    for j in range(size_y):
        y=im_gray[int(im_y_size/size_y*j),int(im_x_size/size_x*i)]
        if y<color_th:
            mat_shape[j,i]=1

#plot用.
mat_test1=copy.copy(mat_shape)

for k in range(len(i_voltage)):
    mat_test1[j_voltage[k],i_voltage[k]]=2

X=np.linspace(0,size_x-1,size_x)
Y=np.linspace(0,-size_y+1,size_y)
fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
plt.pcolormesh(X,Y,mat_test1, cmap="gray")
cbar = plt.colorbar()
plt.show()

PEEC_2d_any_shape_v4_make_matrix_v1.make_matrix(config_csv_file_path)

#メッシュの一区間の辺の長さ.
dx=len_x/(size_x-1)
dy=len_y/(size_y-1)

mat_shape_node_number=np.zeros_like(mat_shape)#何番目のノードかを格納 (0,1,2,3,...).

#mat_shapeが1の箇所の順番を格納.
#座標：(x, y)=(i, j)*(len_x, len_y)
mat_true_i=[]
mat_true_j=[]
k=0
for i in range(size_x):
    for j in range(size_y):
        if mat_shape[j,i]==1:
            mat_true_i.append(i)
            mat_true_j.append(j)
            mat_shape_node_number[j,i]=k
            k+=1
num_true=len(mat_true_i)

#接続行列Aの生成.
#x
Ax=np.zeros([num_true,num_true])
mat_true_x_Ix=[]
mat_true_y_Ix=[]
mat_true_i_Ix=[]
mat_true_j_Ix=[]
k2=0
for k in range(num_true):
    i=mat_true_i[k]
    j=mat_true_j[k]
    if i==size_x-1:
        #右端まで行った. i+1の座標が無くなるためbreak.
        break
    if mat_shape[j,i+1]==1:
        node_num = mat_shape_node_number[j,i]
        Ax[k2,node_num]=-1
        right_node_num = mat_shape_node_number[j,i+1]
        Ax[k2,right_node_num]=1
        mat_true_x_Ix.append(i*dx+dx/2)#ixが流れる平面の中心点の座標.Lxの生成に使う.
        mat_true_y_Ix.append(j*dy)
        mat_true_i_Ix.append(i)#ixの番号.Lxの生成に使う.
        mat_true_j_Ix.append(j)
        k2+=1
Ax=Ax[:k2]

#y
Ay=np.zeros([num_true,num_true])
mat_true_x_Iy=[]
mat_true_y_Iy=[]
mat_true_i_Iy=[]
mat_true_j_Iy=[]
k2=0
for k in range(num_true):
    i=mat_true_i[k]
    j=mat_true_j[k]
    if j==size_y-1:
        #下まで行った. j+1の座標が無くなるためcontinue.
        continue
    if mat_shape[j+1,i]==1:
        node_num = mat_shape_node_number[j,i]
        Ay[k2,node_num]=-1
        down_node_num = mat_shape_node_number[j+1,i]
        Ay[k2,down_node_num]=1
        mat_true_x_Iy.append(i*dy)#iyが流れる平面の中心点の座標.Lyの生成に使う.
        mat_true_y_Iy.append(j*dy+dy/2)
        mat_true_i_Iy.append(i)#iyの番号.Lyの生成に使う.
        mat_true_j_Iy.append(j)
        k2+=1
Ay=Ay[:k2]

with open(csv_folder_path+"\\Rx.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    Rx = np.array(list(reader)).astype(float)
with open(csv_folder_path+"\\Ry.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    Ry = np.array(list(reader)).astype(float)
with open(csv_folder_path+"\\Lx.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    Lx = np.array(list(reader)).astype(float)
with open(csv_folder_path+"\\Ly.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    Ly = np.array(list(reader)).astype(float)
with open(csv_folder_path+"\\Cap.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    Cap = np.array(list(reader)).astype(float)

print("MNA")
start_time = time.time()

#MNA行列の生成.
#ノード名の順番にソート.
mat_boundary=[]
for k in range(len(i_voltage)):
    i=i_voltage[k]
    j=j_voltage[k]
    now=i*size_y+j
    mat_boundary.append([now,i_voltage[k],j_voltage[k],voltage[k]])
mat_boundary_sorted = sorted(mat_boundary, key=lambda x: x[0])
for k in range(len(i_voltage)):
    i_voltage[k]=mat_boundary_sorted[k][1]
    j_voltage[k]=mat_boundary_sorted[k][2]
    voltage[k]=mat_boundary_sorted[k][3]

im_matZ=[]
re_matZ=[]
freq_count=0
for f in matf:
    Omega=2*np.pi*f
    s=1j*Omega
    mat1 = np.hstack((-Ax,-(Rx+s*Lx),np.zeros([len(Rx),len(Ry[0])])))#行方向の結合.
    mat2 = np.hstack((-Ay,np.zeros([len(Ry),len(Rx[0])]),-(Ry+s*Ly)))
    mat3 = np.hstack((s*Cap,-Ax.T,-Ay.T))
    mat_MNA = np.vstack((mat1,mat2))#全て結合.
    mat_MNA = np.vstack((mat_MNA,mat3))

    #電圧の境界条件.
    # mat_MNA * [V;I] = matB.T
    matB=np.zeros(len(mat_MNA[0])-len(i_voltage), dtype=np.complex128)

    k_voltage=[]#i_voltage, j_voltage の座標のノード番号を格納.
    i_voltage=np.concatenate((i_voltage,np.array([-1])))
    j_voltage=np.concatenate((j_voltage,np.array([-1])))
    k2=0
    for k in range(num_true):
        i=mat_true_i[k]
        j=mat_true_j[k]
        if i==i_voltage[k2] and j==j_voltage[k2]:
            k_voltage.append(k)
            k2+=1
    i_voltage=i_voltage[:len(i_voltage)-1]
    j_voltage=j_voltage[:len(j_voltage)-1]

    #行を消す.
    k_zero=len(Ax)+len(Ay)
    k2=0
    for k in k_voltage:
        mat_MNA=np.delete(mat_MNA, k+k_zero-k2, axis=0)
        k2+=1

    #消す列を負にして電圧を掛けてBに入れる.
    k3=0
    for k in k_voltage:
        vol=voltage[k3]
        for k2 in range(len(matB)):
            matB[k2]+=-mat_MNA[k2,k-k3]*vol
        mat_MNA=np.delete(mat_MNA, k-k3, axis=1)#列を消す.
        k3+=1

    #解く.
    #y=np.linalg.solve(mat_MNA, matB)#きっと遅い.
    # mat_MNA を CSC 形式に変換
    mat_MNA_csc = csc_matrix(mat_MNA)
    y=spsolve(mat_MNA_csc, matB)

    end_time = time.time()
    print("f="+"%.1e" % f,end=" ")
    print(f"{end_time - start_time:4f} sec",end=" ")

    #電圧の境界条件を解に戻す.
    k2=0
    for k in k_voltage:
        y=np.concatenate((y[:k],np.array([voltage[k2]]),y[k:]))
        k2+=1

    #ここからデコード.解yから電圧v,電流ix,iyを取り出す.
    v_ans=np.full((size_y,size_x), np.nan, dtype=complex)
    k=0
    for i in range(size_x):
        for j in range(size_y):
            if mat_shape[j,i]==1:
                v_ans[j,i]=y[k]
                k+=1

    ix_ans=np.zeros([size_y,size_x-1], dtype=complex)
    k_zero=k
    mask_num=len(mat_true_x_Ix)
    for k in range(mask_num):
        i=mat_true_i_Ix[k]
        j=mat_true_j_Ix[k]
        ix_ans[j,i]=y[k+k_zero]

    iy_ans=np.zeros([size_y-1,size_x], dtype=complex)
    k_zero+=k+1
    mask_num=len(mat_true_x_Iy)
    for k in range(mask_num):
        i=mat_true_i_Iy[k]
        j=mat_true_j_Iy[k]
        iy_ans[j,i]=y[k+k_zero]

    #ここからインピーダンスの計算.
    #測定する電極から流出する電流を計算.
    measure_current=[]
    measure_voltage=[]
    for k1 in range(len(measure_node)):
        measure_current1=0+1j*0
        measure_voltage1=0+1j*0
        for k2 in range(len(measure_node[0])):
            i=i_voltage[measure_node[k1][k2]]
            j=j_voltage[measure_node[k1][k2]]
            if i==0:
                measure_current_x=ix_ans[j,i]
            elif i==size_x-1:
                measure_current_x=-ix_ans[j,i-1]
            else:
                measure_current_x=ix_ans[j,i]-ix_ans[j,i-1]
            if j==0:
                measure_current_y=iy_ans[j,i]
            elif j==size_y-1:
                measure_current_y=-iy_ans[j-1,i]
            else:
                measure_current_y=iy_ans[j,i]-iy_ans[j-1,i]
            measure_current1+=measure_current_x+measure_current_y
            measure_voltage1+=v_ans[j,i]
        measure_current.append(measure_current1)
        measure_voltage.append(measure_voltage1/len(measure_node[0]))
    
    #print("input voltage=",end="")
    #print(measure_voltage)
    #print("input current=",end="")
    #print(measure_current)
    #インピーダンスを計算.
    for k in range(len(measure_current)):
        impedance=measure_voltage[k]/measure_current[k]
        print(np.real(impedance),np.imag(impedance))
    im_matZ.append(np.imag(impedance))
    re_matZ.append(np.real(impedance))

    #ここから電流の可視化のためのいくつかの演算.
    #ノードの左右を流れる電流の平均を計算.
    ix_ans_node=np.full((size_y,size_x), np.nan, dtype=complex)
    k=0
    for i in range(size_x):
        for j in range(size_y):
            if mat_shape[j,i]==1:
                if i==0:
                    ix_ans_node[j,i]=ix_ans[j,i]
                elif i==size_x-1:
                    ix_ans_node[j,i]=ix_ans[j,i-1]
                else:
                    ix_ans_node[j,i]=(ix_ans[j,i-1]+ix_ans[j,i])/2

    #ノードの上下を流れる電流の平均を計算.
    iy_ans_node=np.full((size_y,size_x), np.nan, dtype=complex)
    k=0
    for i in range(size_x):
        for j in range(size_y):
            if mat_shape[j,i]==1:
                if j==0:
                    iy_ans_node[j,i]=iy_ans[j,i]
                elif j==size_y-1:
                    iy_ans_node[j,i]=iy_ans[j-1,i]
                else:
                    iy_ans_node[j,i]=(iy_ans[j-1,i]+iy_ans[j,i])/2

    #ノードを横切る電流を計算.
    i_ans_node=np.full((size_y,size_x), np.nan)
    i_vec_x=np.full((size_y,size_x), np.nan)
    i_vec_y=np.full((size_y,size_x), np.nan)
    k=0
    for i in range(size_x):
        for j in range(size_y):
            ix=ix_ans_node[j,i]
            iy=iy_ans_node[j,i]
            i_ans_node[j,i]=np.sqrt(np.abs(ix)**2+np.abs(iy)**2)#複素数の絶対値をとり、それぞれのベクトルの絶対値.
            norm_real_i=np.sqrt(np.real(ix)**2+np.real(iy)**2)
            #i_vec_x,i_vec_yは、複素電流の実部のみのベクトルの方向.入力電圧に対する位相を見たいから.plotするとかさばるから、大きさは1.
            i_vec_x[j,i]=np.real(ix)/norm_real_i
            i_vec_y[j,i]=-np.real(iy)/norm_real_i

    #plt.quiver()のための行列.各ベクトルの中心座標を格納.
    X_quiv=np.zeros([size_y,size_x])
    for i in range(size_x):
        for j in range(size_y):
            X_quiv[j,i]=i/(size_x-1)*len_x

    Y_quiv=np.zeros([size_y,size_x])
    for i in range(size_x):
        for j in range(size_y):
            Y_quiv[j,i]=-j/(size_y-1)*len_y

    X=np.linspace(0,len_x,size_x)
    Y=np.linspace(0,-len_y,size_y)
    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
    title_str="%.1e" % f
    plt.title(title_str)
    plt.pcolormesh(X,Y,np.abs(v_ans), cmap="jet")
    cbar = plt.colorbar()
    plt.savefig(fig_folder_path+"\\"+"%03d" % (freq_count+1)+"_voltage.png", format="png", dpi=300)

    X=np.linspace(0,len_x,size_x)
    Y=np.linspace(0,-len_y,size_y)
    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
    title_str="%.1e" % f
    plt.title(title_str)
    pcm = plt.pcolormesh(X, Y, i_ans_node, cmap="jet", norm=LogNorm(vmin=np.nanmin(i_ans_node), vmax=np.nanmax(i_ans_node)))#ログスケール.
    #plt.pcolormesh(X,Y,i_ans_node, cmap="jet")#リニアスケール.
    cbar = plt.colorbar()
    plt.quiver(X_quiv,Y_quiv, i_vec_x, i_vec_y)
    plt.savefig(fig_folder_path+"\\"+"%03d" % (freq_count+1)+"_current.png", format="png", dpi=300)
    #plt.show()
    plt.close('all')
    freq_count+=1

with open(csv_folder_path+"\\freq.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows([matf])

with open(csv_folder_path+"\\im_Z.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows([im_matZ])

with open(csv_folder_path+"\\re_Z.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows([re_matZ])