###HOSVDとHOOIを対戦###


import numpy as np
from scipy import linalg
from scipy.stats import ortho_group
import main


get_np = np.load('../svd/three_eyes.npy')


#もとの評価関数を(27, 27, 27)のテンソルとしてHOOI
def hooi(X, r):
	X = X.reshape(27, 27, 27)
	#任意の直交行列を作る
	L = ortho_group.rvs(27)[:r, :]
	M = ortho_group.rvs(27)[:r, :]
	R = ortho_group.rvs(27)[:r, :]
	for i in range(20):
		LX = np.tensordot(X, L.transpose(), (0,0)).transpose(2,0,1)
		#print(LX.shape) 
		MX = np.tensordot(LX, M.transpose(), (1,0)).transpose(0,2,1)
		#print(MX.shape)
		MX = MX.reshape(r**2, 27)
		U, s, V = linalg.svd(MX)
		R = V[:r, :]

		RX = np.tensordot(X, R.transpose(), (2,0))
		#print(RX.shape)
		LX = np.tensordot(RX, L.transpose(), (0,0)).transpose(1,2,0)
		#print(LX.shape)
		LX = LX.reshape(r**2, 27)
		U, s, V = linalg.svd(LX)
		M = V[:r, :]

		MX = np.tensordot(X, M.transpose(), (1,0)).transpose(0,2,1)
		#print(MX.shape)
		RX = np.tensordot(MX, R.transpose(), (2,0)).transpose(1,2,0)
		#print(RX.shape)
		RX = RX.reshape(r**2, 27)
		U, s, V = linalg.svd(RX)
		L = V[:r, :]
	#コアテンソル
	C2 = np.tensordot(X, L.transpose(), (0,0)).transpose(2,0,1)
	#print(C2.shape)
	C1 = np.tensordot(C2, M.transpose(), (1,0)).transpose(0,2,1)
	#print(C1.shape)
	C = np.tensordot(C1, R.transpose(), (2,0))
	#print(C.shape)
	#復元
	Y2 = np.tensordot(C, L, (0,0)).transpose(2,0,1)
	#print(Y2.shape)
	Y1 = np.tensordot(Y2, M, (1,0)).transpose(0,2,1)
	#print(Y1.shape)
	Y = np.tensordot(Y1, R, (2,0))
	#print(Y.shape)
	#圧縮率
	rate = (L.size + M.size + R.size + C.size) / X.size

	return [Y.reshape((3,3,3,3,3,3,3,3,3)), rate]


#もとの評価関数を(27, 27, 27)のテンソルとしてHOSVD
def hosvd(X, r): 
	X = X.reshape(27, 27, 27)
	#右
	XR = X.reshape(729, 27) 
	_, _, v = linalg.svd(XR)                                              
	VRt = v[:r, :]                                    #vダガー
	VR = np.transpose(VRt)                             #v
	#真ん中
	XM = X.transpose(0, 2, 1)
	XM = XM.reshape(729, 27)
	_, _, v = linalg.svd(XM)                                              
	VMt = v[:r, :]                                    #vダガー
	VM = np.transpose(VMt)                             #v
	#左
	XL = X.transpose(2, 1, 0)
	XL = XL.reshape(729, 27)
	_, _, v = linalg.svd(XL)                                              
	VLt = v[:r, :]                                    #vダガー
	VL = np.transpose(VLt)                             #v
	#コアテンソル
	C2 = np.tensordot(X, VR, (2,0))             
	C1 = np.tensordot(C2, VM, (1,0)) 
	C1 = C1.transpose(0, 2, 1)                      #tensordotの計算によるテンソルの順番を調整
	C = np.tensordot(C1, VL, (0,0))  
	C = C.transpose(2, 0, 1)         
	#復元
	Y2 = np.tensordot(C, VRt, (2,0))  
	Y1 = np.tensordot(Y2, VMt, (1,0))
	Y1 = Y1.transpose(0, 2, 1)
	Y = np.tensordot(Y1, VLt, (0,0))
	Y = Y.transpose(2, 0, 1)  
	#圧縮率
	rate = (VRt.size + VMt.size + VLt.size + C.size) / X.size
	
	return [Y.reshape((3,3,3,3,3,3,3,3,3)), rate]


#HOSVD vs HOOI(同じ圧縮率で対戦)
def cmpr(rate, num_hosvd, num_hooi):  #圧縮率、hosvd特異値数、hooi特異値数
	y1 = main.battle(hooi(get_np, num_hooi)[0], hosvd(get_np, num_hosvd)[0])[0]  #hooiの勝率
	y2 = main.battle(hooi(get_np, num_hooi)[0], hosvd(get_np, num_hosvd)[0])[1]  #hosvdの勝利
	y3 = main.battle(hooi(get_np, num_hooi)[0], hosvd(get_np, num_hosvd)[0])[2]  #引き分け
	return [rate, y1, y2, y3]


#HOSVD vs HOOI
#5回分の平均と標準偏差を算出しdatファイルを作成
def std_calc():
	rate = [0.0, 0.013, 0.027, 0.059, 0.092, 0.14, 0.23, 0.32, 0.56, 0.71, 1.0]
	num_hosvd = [1, 3, 5, 8, 10, 12, 15, 17, 21, 23, 26]
	num_hooi = [1, 3, 5, 8, 10, 12, 15, 17, 21, 23, 26]
	with open("hooi_task2.dat", "w") as f:
		for i, j, k in zip(rate, num_hosvd, num_hooi):
			x = cmpr(i, j, k)[0]   #圧縮率
			y1 = []   #hooi
			y2 = []   #hosvd
			y3 = []   #引き分け
			for _ in range(5):
				y1.append(cmpr(i, j, k)[1])
				y2.append(cmpr(i, j, k)[2])
				y3.append(cmpr(i, j, k)[3])
			y1_m = np.mean(y1)
			y2_m = np.mean(y2)
			y3_m = np.mean(y3)
			y1_std = np.std(y1)
			y2_std = np.std(y2)
			y3_std = np.std(y3)
			f.write("{} {} {} {} {} {} {}\n".format(x, y1_m, y2_m, y3_m, y1_std, y2_std, y3_std))


std_calc()

