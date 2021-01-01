import numpy as np
import random


#win関数
def win(input_list):
    if (input_list[0] == input_list[1] == input_list[2] == 1 or  
        input_list[3] == input_list[4] == input_list[5] == 1 or
        input_list[6] == input_list[7] == input_list[8] == 1 or
        input_list[0] == input_list[3] == input_list[6] == 1 or
        input_list[1] == input_list[4] == input_list[7] == 1 or
        input_list[2] == input_list[5] == input_list[8] == 1 or
        input_list[0] == input_list[4] == input_list[8] == 1 or
        input_list[2] == input_list[4] == input_list[6] == 1):
        return 1       #丸の勝ち
    elif (input_list[0] == input_list[1] == input_list[2] == 2 or  
        input_list[3] == input_list[4] == input_list[5] == 2 or
        input_list[6] == input_list[7] == input_list[8] == 2 or
        input_list[0] == input_list[3] == input_list[6] == 2 or
        input_list[1] == input_list[4] == input_list[7] == 2 or
        input_list[2] == input_list[5] == input_list[8] == 2 or
        input_list[0] == input_list[4] == input_list[8] == 2 or
        input_list[2] == input_list[4] == input_list[6] == 2):
        return -1      #バツの勝ち 
    else:
        return 0       #引き分け


#要素0のインデックス抽出
def func(input_list):
    return [i for i, x in enumerate(input_list) if x == 0]


#盤面を表示
def draw_grid(g2):
    g = [" "]*9
    for i in range(9):
        if g2[i] == 1:
            g[i] = "o"
        elif g2[i] == 2:
            g[i] = "x"
    print("----")
    print(f"{g[0]}{g[1]}{g[2]}")
    print(f"{g[3]}{g[4]}{g[5]}")
    print(f"{g[6]}{g[7]}{g[8]}")


#全探索して評価値を返す(再帰)
a = [[[[[[[[[0 for i9 in range(3)] for i8 in range(3)] for i7 in range(3)] for i6 in range(3)] for i5 in range(3)] for i4 in range(3)] for i3 in range(3)] for i2 in range(3)] for i1 in range(3)]   #9次元配列の作成
def all_search(input_list):
    count0 = input_list.count(0)      #要素0の数
    index0 = func(input_list)         #要素0のインデックスのリスト
    if count0 == 0:                   #終端条件
        return win(input_list)
    if win(input_list) != 0:          #終端条件
        return win(input_list)
    s = 0
    for i in range(count0):                      
        l = index0[i]           #要素0のインデックス
        input_list2 = input_list.copy()  
        input_list2[l] = 2 - (count0 % 2)            #盤面の更新
        s += all_search(input_list2)
    a[input_list[0]][input_list[1]][input_list[2]][input_list[3]][input_list[4]][input_list[5]][input_list[6]][input_list[7]][input_list[8]] = s/count0
    return s / count0


#np配列をファイルに保存
def np_array(input_list):
    all_search(input_list)
    np.save('study/svd_study/three_eyes', np.array(a))        #リストをnp配列に直して保存
    return        


#与えた盤面(入力した文字列)に対する評価値を返す
def Q(input_list, np1):
    return np1[input_list[0]][input_list[1]][input_list[2]][input_list[3]][input_list[4]][input_list[5]][input_list[6]][input_list[7]][input_list[8]]


#softmax関数
def softmax(x):
    a = 10
    u = np.sum(np.exp(a*x))
    return np.exp(a*x) / u


#softmax関数で次の手の評価値を確率化         
def prob(input_list, np1):
    count0 = input_list.count(0)      #要素0の数
    index0 = func(input_list)         #要素0のインデックスのリスト
    s = []
    for i in range(count0):
        l = index0[i]
        input_list2 = input_list.copy()
        input_list2[l] = 2 - (count0 % 2)                 #盤面の更新
        if count0 % 2 == 0:
            s.append(Q(input_list2, np1) * -1)         #次の手がバツのときは評価値にマイナスをつける      
        else:
            s.append(Q(input_list2, np1))
    x = np.array(s)
    y = softmax(x)
    return list(y)


#次の一手を返す   
def next(np1, np2):                       #np1先攻、np2後攻
    input_list = [0]*9
    for i in range(9):
        index0 = func(input_list)    #要素0のインデックスのリスト
        if win(input_list) != 0:
            break
        if (i % 2) == 0:        #先攻
            #print(prob(input_list, np1))
            y = random.choices(index0, k = 1, weights = prob(input_list, np1))[0]
        if (i % 2) == 1:        #後攻
            #print(prob(input_list, np2))
            y = random.choices(index0, k = 1, weights = prob(input_list, np2))[0]
        input_list[y] = (i % 2) + 1            #盤面の更新
    return win(input_list)


#先行後攻の有利不利を加味した結果を返す関数
def battle(np1, np2):
    w1 = 0           #np1の勝ち
    w2 = 0           #np2の勝ち
    trial = 500
    for _ in range(trial):
        w = next(np1, np2)
        if w == 1:    #np1の勝ち
            w1 += 1
        if w == -1:   #np2の勝ち
            w2 += 1  
        w = next(np2, np1)
        if w == 1:    #np2の勝ち
            w2 += 1
        if w == -1:   #np1の勝ち
            w1 += 1
    w1_ratio = w1 / (trial*2)
    w2_ratio = w2 / (trial*2)
    return [w1_ratio, w2_ratio, 1 - w1_ratio - w2_ratio]




#実行
s = "000000000"
list_0 = list(map(int, list(s)))   

#np_array(list_0)   #np配列をファイルに保存

#draw_grid(list_0)
