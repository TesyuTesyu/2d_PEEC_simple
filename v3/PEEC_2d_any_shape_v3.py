import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import time
import copy
from matplotlib.colors import LogNorm

'''
・遅延を考慮していない (R, L, C) PEEC。
・自由空間における導体版のシミュレーション。
・形状は画像から生成される。黒色が導体の領域、白色がなにもない領域となる。
・mutual L.Cは全面積で計算されている.
・本来、物体の輪郭の微小面は面積を半分にするが、このプログラムは全て同じ面積で処理されている。
・python 3.12.1 VS-codeで動作.

参考：
[1] Ekman, Jonas, "Electromagnetic modeling using the partial element equivalent circuit method," Luleå tekniska universitet,  Doctoral thesis, 2003.

X: @Testes_int
'''

def rectangular_solid_self_inductance(width,length,tickness,mu):
    #[1] p99-100
    u=length/width
    Omega=tickness/width
    A1=np.sqrt(1+u**2)
    A2=np.sqrt(1+Omega**2)
    A3=np.sqrt(u**2+Omega**2)
    A4=np.sqrt(u**2+Omega**2+1)
    A5=np.log((1+A4)/A3)
    A6=np.log((Omega+A4)/A1)
    A7=np.log((u+A4)/A2)

    Lp=length*2*mu/np.pi*(Omega**2/24/u*(np.log((1+A2)/Omega)-A5)+1/24/u/Omega*(np.log(Omega+A2)-A6)+Omega**2/60/u*(A4-A3)+Omega**2/24*(np.log((u+A3)/Omega)-A7)+Omega**2/60/u*(Omega-A2)+1/20/u*(A2-A4)+u/4*A5-u**2/6/Omega*np.arctan(Omega/u/A4)+u/4/Omega*A6-Omega/6*np.arctan(u/Omega/A4)+A7/4-1/6/Omega*np.arctan(Omega*u/A4)+1/24/Omega**2*(np.log(u+A1)-A7)+u/20/Omega**2*(A1-A4)+1/60/Omega**2/u*(1-A2)+1/60/u/Omega**2*(A4-A1)+u/20*(A3-A4)+u**3/24/Omega**2*(np.log((1+A1)/u)-A5)+u**3/24/Omega*(np.log((Omega+A3)/u)-A6)+u**3/60/Omega**2*((A4-A1)+(u-A3)))
    return Lp

def rectangular_solid_mutual_inductance(width1,width2,length1,length2,Dx,Dy,mu):
    #[1] p99-100. ただしミスなのか、Dx, Dyが逆になっている.引用元の論文を確認したがそちらは正しかったため、おそらくこれで正しい.
    a=[Dy-length1/2-length2/2,Dy+length1/2-length2/2,Dy+length1/2+length2/2,Dy-length1/2+length2/2]
    b=[Dx-width1/2-width2/2,Dx+width1/2-width2/2,Dx+width1/2+width2/2,Dx-width1/2+width2/2]
    matinteg2=[]
    for i in range(4):
        matinteg=[]
        for j in range(4):
            rho=np.sqrt(a[i]**2+b[j]**2)
            matinteg.append((-1)**(i+j+2)*(b[j]**2*a[i]/2*np.log(a[i]+rho)+a[i]**2*b[j]/2*np.log(b[j]+rho)-rho/6*(b[j]**2+a[i]**2)))
        matinteg2.append(sum(matinteg))
    integ=sum(matinteg2)
    Lpm=mu/np.pi/4/(width1*width2)*integ
    return Lpm

def rectangular_solid_self_potential(width,length,eps):
    #[1] p61
    u=length/width
    p=length/4/np.pi/eps*2/3*(3*np.log(u+np.sqrt(u**2+1))+u**2+1/u+3*u*np.log(1/u+np.sqrt(1/u**2+1))-(np.sqrt(u**(4/3)+(1/u)**(2/3)))**3)
    return p

def rectangular_solid_mutual_potential(width1,width2,length1,length2,Dx,Dy,eps):
    #[1] p62
    a=[Dx-width1/2-width2/2,Dx+width1/2-width2/2,Dx+width1/2+width2/2,Dx-width1/2+width2/2]
    b=[Dy-length1/2-length2/2,Dy+length1/2-length2/2,Dy+length1/2+length2/2,Dy-length1/2+length2/2]
    matinteg2=[]
    for i in range(4):
        matinteg=[]
        for j in range(4):
            rho=np.sqrt(a[i]**2+b[j]**2)
            matinteg.append((-1)**(i+j+2)*(b[j]**2*a[i]/2*np.log(a[i]+rho)+a[i]**2*b[j]/2*np.log(b[j]+rho)-rho/6*(b[j]**2+a[i]**2)))
        matinteg2.append(sum(matinteg))
    integ=sum(matinteg2)
    p=1/4/np.pi/eps/(width1*width2*length1*length2)*integ
    return p

#---------------------定数の設定はここから.--------------------

# 真空の透磁率、誘電率、導体の抵抗率.
mu=4*np.pi*1e-7
eps= 8.854e-12
rho_ohm=1.68e-8

len_x=100e-3 #横方向の長さ[m].縦は画像のアスペクトで決まる.
size_x=50 #横方向の分割数.縦は画像のアスペクトで決まる.

tickness=0.01e-3 #導体面の厚さ[m].

f=100e3 #周波数[Hz].
#cut_off_radius=0.1#mutual L,C を計算する範囲.円形で、この値は半径.未実装.

#画像の設定
color_th=100 #白か黒かをこの閾値で判定.
file_path=r"/??.csv" #ファイルパス.

#電圧源の位置.
i_voltage=[1,49] #x方向の位置(?番目).
j_voltage=[18,18] #y方向の位置(?番目).
voltage=[1,0] #それぞれの電圧.
measure_node=[[0]]

gjkl=37
i_voltage=np.concatenate([np.zeros(gjkl,dtype="int"),49*np.ones(gjkl,dtype="int")])
j_voltage=np.concatenate([np.linspace(0,gjkl-1,gjkl,dtype="int"),np.linspace(0,gjkl-1,gjkl,dtype="int")])
voltage=np.concatenate([np.ones(gjkl,dtype="int"),np.zeros(gjkl,dtype="int")])
measure_node=[np.linspace(0,gjkl-1,gjkl,dtype="int")]#インピーダンスの測定に用いるノード.

'''
gjkl=18
i_voltage=np.concatenate([np.zeros(gjkl,dtype="int"),24*np.ones(gjkl,dtype="int")])
j_voltage=np.concatenate([np.linspace(0,gjkl-1,gjkl,dtype="int"),np.linspace(0,gjkl-1,gjkl,dtype="int")])
voltage=np.concatenate([np.ones(gjkl,dtype="int"),np.zeros(gjkl,dtype="int")])
measure_node=[np.linspace(0,gjkl-1,gjkl,dtype="int")]#インピーダンスの測定に用いるノード.
'''
#---------------------定数の設定はここまで.--------------------

'''
#カウント変数たち.
iはx方向 (0 < i < size_x)
jはy方向 (0 < j < size_y)
kはノード番号や面の番号 (0 < k < size_x*size_y) or (0 < k < (size_x-1)*size_y) or (0 < k < size_x*(size_y-1))
'''

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

'''
#形状 (1がTrue).デバッグ用.
mat_shape=[[1,1,1,1],[1,1,1,1],[0,1,0,1],[0,0,1,1],[1,1,1,1]]
mat_shape=np.array(mat_shape)
size_x=len(mat_shape[0])
size_y=len(mat_shape)

i_voltage=[0,3,-1]
j_voltage=[0,4,-1]
voltage=[0,1,0]
'''
'''
print("[",end="")
for j in range(size_y):
    for i in range(size_x):
        print(mat_shape[j,i],end="")
        if i!=size_x-1:
            print(",",end="")
    print("],[",end="")
'''

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
print("A",end="")
start_time = time.time()
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

end_time = time.time()
print(f": {end_time - start_time:4f} sec")

#Lx,Lyの生成.
#まず、座標(0,0)、全てのメッシュ点間の相互L（(0,0)対(0,0)は自己L）を計算し格納する.
print("L",end="")
start_time = time.time()

#x
#x方向のLx (k1, k2)
Lp_x=rectangular_solid_self_inductance(dy,dx,tickness,mu)

matx=np.linspace(dx/2,len_x-dx/2,size_x-1)
maty=np.linspace(0,len_y,size_y)

matLx_cash=np.zeros([size_y,size_x-1])
#(0,0)と(i,j)間のLを格納.
k1=0
for i in range(size_x-1):#x方向の移動.
    x=matx[i]
    for j in range(size_y):#y方向の移動.
        if i==0 and j==0:
            matLx_cash[0,0]=Lp_x
        elif (i==1 and j==0) or (i==1 and j==1) or (i==0 and j==1):
            #(0,0)の周囲は計算できない.
            matLx_cash[j,i]=0
        else:
            y=maty[j]
            distance_x=np.abs(y)#ここx, yが逆. 理由は定義を参照.
            distance_y=np.abs(x-dx/2)
            matLx_cash[j,i]=rectangular_solid_mutual_inductance(dy,dy,dx,dx,distance_x,distance_y,mu)

mask_num=len(mat_true_x_Ix)
Lx=np.zeros([mask_num,mask_num])
for k1 in range(mask_num):
    #面1のx,y 中心座標.
    i1=mat_true_i_Ix[k1]
    j1=mat_true_j_Ix[k1]
    for k2 in range(mask_num):
        #面2のx,y 中心座標.
        i2=mat_true_i_Ix[k2]
        j2=mat_true_j_Ix[k2]
        Lx[k1,k2]=matLx_cash[abs(j2-j1),abs(i2-i1)]


#x
#x方向のLx (k1, k2)
Lp_y=rectangular_solid_self_inductance(dx,dy,tickness,mu)

matx=np.linspace(0,len_x,size_x)
maty=np.linspace(dy/2,len_y-dy/2,size_y-1)

matLy_cash=np.zeros([size_y-1,size_x])
#(0,0)と(i,j)間のLを格納.
k1=0
for i in range(size_x):#x方向の移動.
    x=matx[i]
    for j in range(size_y-1):#y方向の移動.
        if i==0 and j==0:
            matLy_cash[0,0]=Lp_y
        elif (i==1 and j==0) or (i==1 and j==1) or (i==0 and j==1):
            #(0,0)の周囲は計算できない.
            matLy_cash[j,i]=0
        else:
            y=maty[j]
            distance_x=np.abs(x)#ここx, yが逆. 理由は定義を参照.
            distance_y=np.abs(y-dy/2)
            matLy_cash[j,i]=rectangular_solid_mutual_inductance(dx,dx,dy,dy,distance_x,distance_y,mu)

mask_num=len(mat_true_x_Iy)
Ly=np.zeros([mask_num,mask_num])
for k1 in range(mask_num):
    #面1のx,y 中心座標.
    i1=mat_true_i_Iy[k1]
    j1=mat_true_j_Iy[k1]
    for k2 in range(mask_num):
        #面2のx,y 中心座標.
        i2=mat_true_i_Iy[k2]
        j2=mat_true_j_Iy[k2]
        Ly[k1,k2]=matLy_cash[abs(j2-j1),abs(i2-i1)]

end_time = time.time()
print(f": {end_time - start_time:4f} sec")

#Rx,Ryの生成.今は全て同じ抵抗で計算してる.
print("R",end="")
start_time = time.time()
Rx=np.zeros([len(Ax),len(Ax)])
Ry=np.zeros([len(Ay),len(Ay)])

#x
Rp_x=rho_ohm*dx/(dy*tickness)
for i in range(len(Ax)):
    Rx[i,i]=Rp_x

#y
Rp_y=rho_ohm*dy/(dx*tickness)
for i in range(len(Ay)):
    Ry[i,i]=Rp_y
end_time = time.time()
print(f": {end_time - start_time:4f} sec")

#Capの生成.方法はLx,Lyとほぼ同じ.
print("C",end="")
start_time = time.time()

Cap=np.zeros([len(Ax[0]),len(Ax[0])])
Cp=1/rectangular_solid_self_potential(dx,dy,eps)#自己インダクタンス.

matx=np.linspace(0,len_x,size_x)
maty=np.linspace(0,len_y,size_y)

matC_cash=np.zeros([size_y,size_x])
#(0,0)と(i,j)間のCを格納.
k1=0
for i in range(size_x):#x方向の移動.
    x=matx[i]
    for j in range(size_y):#y方向の移動.
        if i==0 and j==0:
            matC_cash[0,0]=Cp
        elif (i==1 and j==0) or (i==1 and j==1) or (i==0 and j==1):
            #(0,0)の周囲は計算できない.
            matC_cash[j,i]=0
        else:
            y=maty[j]
            distance_x=np.abs(y)#ここx, yが逆. 理由は定義を参照.
            distance_y=np.abs(x)
            matC_cash[j,i]=1/rectangular_solid_mutual_potential(dx,dx,dy,dy,distance_x,distance_y,eps)

for k1 in range(num_true):
    i1=mat_true_i[k1]
    j1=mat_true_j[k1]
    for k2 in range(num_true):
        i2=mat_true_i[k2]
        j2=mat_true_j[k2]
        Cap[k1,k2]=matC_cash[abs(j2-j1),abs(i2-i1)]

#作りかけ.

end_time = time.time()
print(f": {end_time - start_time:4f} sec")

print("MNA",end="")
start_time = time.time()

#MNA行列の生成. A,R,L,C行列はメモリの削減のためにここで消される.
k_zero=len(Ax)+len(Ay)

Omega=2*np.pi*f
s=1j*Omega
mat1 = np.hstack((-Ax,-(Rx+s*Lx),np.zeros([len(Rx),len(Ry[0])])))#行方向の結合.
mat2 = np.hstack((-Ay,np.zeros([len(Ry),len(Rx[0])]),-(Ry+s*Ly)))
Rx=[]
Lx=[]
Ry=[]
Ly=[]
mat3 = np.hstack((s*Cap,-Ax.T,-Ay.T))
Ax=[]
Ay=[]
Cap=[]
mat_MNA = np.vstack((mat1,mat2))#全て結合.
mat1=[]
mat2=[]
mat_MNA = np.vstack((mat_MNA,mat3))

#電圧の境界条件.
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

#行を消す.
k2=0
#k_zero=len(Ax)+len(Ay). この段階でAxは消えてるので消す前の場所に持って行った.
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
y=spsolve(mat_MNA, matB)

end_time = time.time()
print(f": {end_time - start_time:4f} sec")

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
print("input voltage=",end="")
print(measure_voltage)
print("input current=",end="")
print(measure_current)
#インピーダンスを計算.
for k in range(len(measure_current)):
    impedance=measure_voltage[k]/measure_current[k]
    print(np.real(impedance),np.imag(impedance))

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
plt.pcolormesh(X,Y,np.abs(v_ans), cmap="jet")
cbar = plt.colorbar()

X=np.linspace(0,len_x,size_x)
Y=np.linspace(0,-len_y,size_y)
fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
pcm = plt.pcolormesh(X, Y, i_ans_node, cmap="jet", norm=LogNorm(vmin=np.nanmin(i_ans_node), vmax=np.nanmax(i_ans_node)))#ログスケール.
#plt.pcolormesh(X,Y,i_ans_node, cmap="jet")#リニアスケール.
cbar = plt.colorbar()
plt.quiver(X_quiv,Y_quiv, i_vec_x, i_vec_y)

'''
X=np.linspace(0,len_x,size_x-1)
Y=np.linspace(0,-len_y,size_y)
fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
plt.pcolormesh(X,Y,np.abs(ix_ans), cmap="jet")
cbar = plt.colorbar()

X=np.linspace(0,len_x,size_x)
Y=np.linspace(0,-len_y,size_y-1)
fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
plt.pcolormesh(X,Y,np.abs(iy_ans), cmap="jet")
cbar = plt.colorbar()
'''
plt.show()