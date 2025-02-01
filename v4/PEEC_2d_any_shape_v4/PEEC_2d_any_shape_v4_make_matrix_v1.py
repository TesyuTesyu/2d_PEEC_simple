import numpy as np
from PIL import Image
import time
import csv

'''
A,R,L,C行列を生成し、CSVで保存する。f特を見るために何度も生成しなおすのは無駄なため.
'''
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

def make_matrix(config_csv_file_path):
    #Ax,Ay,Lx,Ly,Cap行列を作る.
    #config.csvから変数を読み出す.
    variables = {}
    with open(config_csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            variable, value = row
            variables[variable] = value

    # 読み出した変数を格納.
    mu = float(variables['mu'])
    eps = float(variables['eps'])
    rho_ohm = float(variables['rho_ohm'])
    len_x = float(variables['len_x'])
    size_x = int(variables['size_x'])
    tickness = float(variables['tickness'])
    color_th = int(variables['color_th'])
    file_path = str(variables['file_path'])
    csv_file_path = str(variables['csv_file_path'])

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
    print("p",end="")
    start_time = time.time()

    potential=np.zeros([len(Ax[0]),len(Ax[0])])
    potential_p=rectangular_solid_self_potential(dx,dy,eps)#自己ポテンシャル.#2025/0202に修正.間違っていた.

    matx=np.linspace(0,len_x,size_x)
    maty=np.linspace(0,len_y,size_y)

    potential_cash=np.zeros([size_y,size_x])
    #(0,0)と(i,j)間のCを格納.
    k1=0
    for i in range(size_x):#x方向の移動.
        x=matx[i]
        for j in range(size_y):#y方向の移動.
            if i==0 and j==0:
                potential_cash[0,0]=potential_p
            elif (i==1 and j==0) or (i==1 and j==1) or (i==0 and j==1):
                #(0,0)の周囲は計算できない.
                potential_cash[j,i]=0
            else:
                y=maty[j]
                distance_x=np.abs(y)#ここx, yが逆. 理由は定義を参照.
                distance_y=np.abs(x)
                potential_cash[j,i]=rectangular_solid_mutual_potential(dx,dx,dy,dy,distance_x,distance_y,eps)

    for k1 in range(num_true):
        i1=mat_true_i[k1]
        j1=mat_true_j[k1]
        for k2 in range(num_true):
            i2=mat_true_i[k2]
            j2=mat_true_j[k2]
            potential[k1,k2]=potential_cash[abs(j2-j1),abs(i2-i1)]

    end_time = time.time()
    print(f": {end_time - start_time:4f} sec")

    print("C",end="")
    start_time = time.time()
    Cap = np.linalg.inv(potential)
    print(f": {end_time - start_time:4f} sec")

    with open(csv_file_path+"\\Rx.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Rx)
    with open(csv_file_path+"\\Ry.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Ry)
    with open(csv_file_path+"\\Lx.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Lx)
    with open(csv_file_path+"\\Ly.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Ly)
    with open(csv_file_path+"\\Cap.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Cap)
    
    return 1
