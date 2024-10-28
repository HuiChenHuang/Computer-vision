# Computer-vision
W1,3,6,7 總整理
https://ithelp.ithome.com.tw/articles/10296810
# Week1:電腦視覺
### 電腦視覺:
是一個**跨學科**的研究領域-從數字圖像或視頻中解釋和理解視覺信息的能力
### 電腦視覺任務:
1. 圖像分類 (Image Classification)
2. 物件定位 (Object Localization)
3 物件偵測 (Object Detection)
4. 圖像分割 (Segmentation)

### 傳統電腦視覺VS現代電腦視覺
![https://ithelp.ithome.com.tw/upload/images/20241025/20151681W54275mQsn.png](https://ithelp.ithome.com.tw/upload/images/20241025/20151681W54275mQsn.png)

# Week2:視覺機器學習模型
### 為什麼電腦視覺很難？
| 1. 物件 | 2. 類別 | 3. 深度 | 4. 相機和感測器性能 | 5. 角度 | 6. 亮度 | 7. 規模 | 8. 動作 | 9. 雜亂(Clutter) | 10. 型態 | 11. 視錯覺 (optical illusions) |
|------|------|------|----------------|------|------|------|------|------------------|------|--------------------------|
##### RGB頻道(channel) 各由多個值為0-255之間的像素(pixel)組成
![https://ithelp.ithome.com.tw/upload/images/20241025/201516812OPMyl2y6v.png](https://ithelp.ithome.com.tw/upload/images/20241025/201516812OPMyl2y6v.png)
### Different task of ML:
##### Supervised Learning (target->error; output: mapping), Unsupervised Learning (output: Classes), Reinforcement Learning (evaluation -> Reward; output:Action)

## 1. SVM (監督式學習,分類)
### 找到一條“最佳超平面”來區分不同類別的數據
### 最大化不同類別距離超平面的間隔(Margin)
![https://ithelp.ithome.com.tw/upload/images/20241025/20151681d97g9anE2N.png](https://ithelp.ithome.com.tw/upload/images/20241025/20151681d97g9anE2N.png)
### 如果資料無法用一條線切開 -> 使用 "過Kernel Trick" 資料到更高維的空間
![https://ithelp.ithome.com.tw/upload/images/20241025/201516810rWlZVeS8t.png](https://ithelp.ithome.com.tw/upload/images/20241025/201516810rWlZVeS8t.png)

### SVM (監督式學習) X Computer Vision 
1. 圖像`分類`
2. 物件`偵測` (分類)
3. 語義`分割` (圖相似的部分做切分)

### SVM 缺點:
 1. 不適合大資料集(訓練耗時)
 2. 非線性資料的處理有限(雖然Kernel Trick能夠擴展 SVM 處理非線性問題, 難以選擇合適的Kernel function以及調整參數)
 3. 多分類問題較複雜 (SVM 原本是二分類, “一對一”或“一對多”增加計算的複雜性)

## 2. KNN (監督式學習, 分類) :距離來判斷資料的類別
![https://ithelp.ithome.com.tw/upload/images/20241028/20151681sxLgOSMogL.png](https://ithelp.ithome.com.tw/upload/images/20241028/20151681sxLgOSMogL.png)

若結果為各半, 判給離X較近的類別(為避免這個情況,K通常會設為奇數)

### 距離method:
1. 歐基里德距離
2. 曼哈頓距離
3. 明氏距離

## KNN 缺點
● 計算效率低
● 在高維度資料下表現不佳
● 無法處理非線性關係

## PCA (降維: 二維 ---> 一維 (新空間（稱為主成分, 是資料中變異性最大的方向）))
● 保持資料主要信息的同時，減少特徵數量

## PCA X Computer Vision
1.圖像降維 (不需要考慮圖像色彩 ex.邊緣偵測), (灰階圖像可以減少計算量，加快訓練速度)
2.圖像去噪 

## PCA 缺點 (線性降維技術)
● 對非線性資料的處理有限
● 訊息遺失 (判斷細節與特徵表現不佳(降維時減少了資料的維度))

## Activation function

Sigmoid: 0 ~ 1

tanh: -1 ~ 1

RELU: 0 ~ 10

Leaky ReLU: -1 ~ 10

ELU: -2 ~ 10

## Classification task
● 二元分類 -> Sigmoid 
● 多類別分類 -> Softmax

## 訓練損失 (training loss) : 
衡量神經網路`預測結果與真實值的差距`，並根據結果來調整神經網路的`權重`
- 分類任務： 交叉熵 (cross-entropy)
- 迴歸任務：Mean squared error (MSE)

### Gradient Descent
● 根據Loss的梯度優化(更新)神經網路的權重，讓Loss最小化

## Learning rate (要調整的Hyperparameter)
● 梯度下降, 權重更新多少?

● 太大 >> 跳過最佳解

● 太小 >> 無法到達最佳解

## 優化器 (Optimizer): 根據訓練損失調整模型參數
- 隨機梯度下降 (SGD): 不穩定性可能會幫助模型脫離local minima
- 自適應梯度 (Adam): 結合了動量（Momentum）和 RMSProp 的優點; 自適應地調整Learning rate

## RMSProps : 參數更新的平方梯度均值來調整Learning rate
● 梯度變化較大 -> 減少Learning rate，防止步伐過大跳過最佳解
● 梯度變化較小 -> 增大學習率，加速收斂

## Metrics
### ● Precision (精確率): True positive (猜對真實也對 / 真實有錯也有對)
### ● Recall (召回率) : 實際為positive的結果 (有猜對, 也有猜錯)
### ● Accuracy (準確率)：(所有猜對的部分: True negaative, True Positive)
![https://ithelp.ithome.com.tw/upload/images/20241028/20151681OgRrHmxkeQ.png](https://ithelp.ithome.com.tw/upload/images/20241028/20151681OgRrHmxkeQ.png)

## 需要設定的超參數
---
*- 每一層*
    - Number of nodes
    - Activation function
- Optimizer
- Learning rate
- Training loss
- Metrics
---
 
# Week 3:



