###もとの評価関数を(27, 27, 27)のテンソルとしてHOOI###


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
#		print(LX.shape) 
		MX = np.tensordot(LX, M.transpose(), (1,0)).transpose(0,2,1)
#		print(MX.shape)
		MX = MX.reshape(r**2, 27)
		U, s, V = linalg.svd(MX)
		R = V[:r, :]

		RX = np.tensordot(X, R.transpose(), (2,0))
#		print(RX.shape)
		LX = np.tensordot(RX, L.transpose(), (0,0)).transpose(1,2,0)
#		print(LX.shape)
		LX = LX.reshape(r**2, 27)
		U, s, V = linalg.svd(LX)
		M = V[:r, :]

		MX = np.tensordot(X, M.transpose(), (1,0)).transpose(0,2,1)
#		print(MX.shape)
		RX = np.tensordot(MX, R.transpose(), (2,0)).transpose(1,2,0)
#		print(RX.shape)
		RX = RX.reshape(r**2, 27)
		U, s, V = linalg.svd(RX)
		L = V[:r, :]
	#コアテンソル
	C2 = np.tensordot(X, L.transpose(), (0,0)).transpose(2,0,1)
#	print(C2.shape)
	C1 = np.tensordot(C2, M.transpose(), (1,0)).transpose(0,2,1)
#	print(C1.shape)
	C = np.tensordot(C1, R.transpose(), (2,0))
#	print(C.shape)
	#復元
	Y2 = np.tensordot(C, L, (0,0)).transpose(2,0,1)
#	print(Y2.shape)
	Y1 = np.tensordot(Y2, M, (1,0)).transpose(0,2,1)
#	print(Y1.shape)
	Y = np.tensordot(Y1, R, (2,0))
#	print(Y.shape)
	#圧縮率
	rate = (L.size + M.size + R.size + C.size) / X.size
	#フロべニウスノルムの相対誤差
	norm = np.sqrt(np.sum(X*X))
	norm1 = np.sqrt(np.sum((X-Y) * (X-Y)))
	frob = norm1 / norm

	return [Y.reshape((3,3,3,3,3,3,3,3,3)), rate, frob]



#print(hooi(get_np, 5))



#5回分の平均と標準偏差を算出しdatファイルを作成
def save_file():
	with open("hooi_task1.dat", "w") as f:
		for i in range(1, 28):
			#圧縮率
			x = hooi(get_np, i)[1]   
			#battle
			y1 = []
			y2 = []
			y3 = []
			for _ in range(5):
				y1.append(main.battle(get_np, hooi(get_np, i)[0])[0])   #originalが勝つ割合 
				y2.append(main.battle(get_np, hooi(get_np, i)[0])[1])   #hooiが勝つ割合
				y3.append(main.battle(get_np, hooi(get_np, i)[0])[2])   #引き分けの割合 
			y1_m = np.mean(y1)
			y2_m = np.mean(y2)
			y3_m = np.mean(y3)
			y1_std = np.std(y1)
			y2_std = np.std(y2)
			y3_std = np.std(y3)
			#frobenius
			y4 = hooi(get_np, i)[2] 
			f.write("{} {} {} {} {} {} {} {}\n".format(x, y1_m, y2_m, y3_m, y4, y1_std, y2_std, y3_std))


save_file()
