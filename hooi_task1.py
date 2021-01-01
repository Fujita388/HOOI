###もとの配列をHOOI###


import numpy as np
from scipy import linalg
from scipy.stats import ortho_group
import main


get_np = np.load('../svd/three_eyes.npy')



def hooi(X, r):
	X = X.reshape(27, 27, 27)
	#任意の直交行列を作る
	L = ortho_group.rvs(27)[:r, :]
	M = ortho_group.rvs(27)[:r, :]
	R = ortho_group.rvs(27)[:r, :]
	for i in range(1):
		LX = np.tensordot(X, L.transpose(), (0,0)).transpose(2,0,1)
		print(LX.shape) 
		MX = np.tensordot(LX, M.transpose(), (1,0)).transpose(0,2,1)
		print(MX.shape)
		MX = MX.reshape(r**2, 27)
		U, s, V = linalg.svd(MX)
		R = V[:r, :]

		RX = np.tensordot(X, R.transpose(), (2,0))
		print(RX.shape)
		LX = np.tensordot(RX, L.transpose(), (0,0)).transpose(1,2,0)
		print(LX.shape)
		LX = LX.reshape(r**2, 27)
		U, s, V = linalg.svd(LX)
		M = V[:r, :]

		MX = np.tensordot(X, M.transpose(), (1,0)).transpose(0,2,1)
		print(MX.shape)
		RX = np.tensordot(MX, R.transpose(), (2,0)).transpose(1,2,0)
		print(RX.shape)
		RX = RX.reshape(r**2, 27)
		U, s, V = linalg.svd(RX)
		L = V[:r, :]
#コアテンソル
	C2 = np.tensordot(X, L.transpose(), (0,0)).transpose(2,0,1)
	print(C2.shape)
	C1 = np.tensordot(C2, M.transpose(), (1,0)).transpose(0,2,1)
	print(C1.shape)
	C = np.tensordot(C1, R.transpose(), (2,0))
	print(C.shape)
#復元
	Y2 = np.tensordot(C, L, (0,0)).transpose(2,0,1)
	print(Y2.shape)
	Y1 = np.tensordot(Y2, M, (1,0)).transpose(0,2,1)
	print(Y1.shape)
	Y = np.tensordot(Y1, R, (2,0))
	print(Y.shape)
#圧縮率
	rate = (L.size + M.size + R.size + C.size) / X.size
#フロべニウスノルムの相対誤差
	norm = np.sqrt(np.sum(X*X))
	norm1 = np.sqrt(np.sum((X-Y) * (X-Y)))
	frob = norm1 / norm

	return [rate, frob]



print(hooi(get_np, 5))

