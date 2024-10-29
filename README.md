# Computer-vision
W1,3,6,7 總整理
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
![https://ithelp.ithome.com.tw/upload/images/20241028/20151681rC2He86XSm.png](https://ithelp.ithome.com.tw/upload/images/20241028/20151681rC2He86XSm.png)

#  Week 6:
## 基本的評估指標
● Training Loss

● Training Accuracy 

● Test/Validation Loss

● Test/Validation Accuracy

## Confusion Matrix 容易預測錯誤的類別 (etc. 3 預測錯為 8)
![https://ithelp.ithome.com.tw/upload/images/20241028/20151681uHVrG4fawf.png](https://ithelp.ithome.com.tw/upload/images/20241028/20151681uHVrG4fawf.png)
![https://ithelp.ithome.com.tw/upload/images/20241028/20151681eDEljYn1gv.png](https://ithelp.ithome.com.tw/upload/images/20241028/20151681eDEljYn1gv.png)
![https://ithelp.ithome.com.tw/upload/images/20241028/201516812sB09W2mHR.png](https://ithelp.ithome.com.tw/upload/images/20241028/201516812sB09W2mHR.png)
![https://ithelp.ithome.com.tw/upload/images/20241028/20151681H5s6jXvNJF.png](https://ithelp.ithome.com.tw/upload/images/20241028/20151681H5s6jXvNJF.png)

## Overfitting
### 降低Overfitting以及增加Generalization (Generalization代表模型在沒見過的資料上的表現)
### 原因: 過度訓練、訓練資料太少、使用太多Features、或使用太複雜的模型
### 解決辦法: 使用Regularization 限制模型的複雜度，讓模型學習正確的特徵, `增加訓練資料`或`降低模型複雜度`
○ 更多資料？

○ 更大更深的模型？

○ 更多Epochs?

## Regularization Methods (LLDDEB)
● L1 & L2 Regularization

● Drop Out

● Data Augmentation

● Early Stopping

● Batch Normalization

## L1 and L2 Regularization 
● 使模型使用較小的參數 (weights and biases)

● 避免特定的節點產生過大的影響

`Loss Function + Penalty (L1 or L2)`

## Differnce between L1與L2
● L1 : 優先將`不重要的特徵的權重降為0` (特徵選取)

● L2 : 優先將所有的`權重減小` (大的權重會產生更大的Penalty)

***L1和L2都會使模型使用較小的權重***

## Dropout (學習更可靠的特徵)
● 在訓練過程中，隨機關閉某一層的一些節點

● 測試模型時，會使用所有的節點來做預測

## Overfitting的原因：沒有足夠的訓練資料 => 使用Data Augmentation可以解決這個問題 (Keras [Callback] 和 Pytorch [手動實作])
○ 翻轉 (水平/垂直)   ○ 亮度和對比   ○ 旋轉

○ 縮放              ○ 裁切         ○ 扭曲

## Early Stopping
設定一個threshold，告訴模型如果在接下來的P 個Epochs (each epoch willsave the weight )都沒有更好的結果，就停止訓練

## Batch Normalization怎麼運作
● each Mini-Batch計算`mean`和`STD`以及進行`標準化`

● 假如有使用Dropout，層的順序為：

○ Conv -> BacthNorm -> ReLU -> Dropout

## Regularization小建議
● 不要一開始就使用Regularization (建立Baseline Model)

● Dropout和Batch Norm會`增加訓練時間`

● Dropout: `不要在Softmax Function 之前使用`

● 很`簡單`的`模型``不太需要使用Regularization`

● 更多Epochs

● 如果`L2 Penalty太高`，模型可能會`Underfitting`

# Week 7: 進階影像視覺模型
## LeNet:
2 conv. layers & max pool > 2nd max pool > flatten > 120 FC layer > 84 FC layer > output

## AlexNet :
● 共有 8 層：前 5 層為Convolutional Layers，後 3 層為 FC Layers
![https://ithelp.ithome.com.tw/upload/images/20241028/201516815xtKMv8qOu.png](https://ithelp.ithome.com.tw/upload/images/20241028/201516815xtKMv8qOu.png)

## VGGNet : “典型CNN” 結構 (龐大的參數量, High Accuracy, slow to train)

● Top-5 Accuracy in ImageNet (1000 classes)

● 多個 Conv Layers 後接 Pooling Layer

● Filters/Feature Maps 的數量逐漸增加，直到 FC Layers

### ● `VGG16` has `13` Conv Layers with `3` FC Layers
### ● `VGG19` has `16` Conv Layers with 3` FC Layers
| **VGGNet 優缺點** |                          |
|------------------|--------------------------|
| **優點**         | - “深度”網路設計           |
|                  | - 規範的結構               |
| **缺點**         | - 計算需求大               |
|                  | - 參數冗餘                 |

## ResNet : (一般的CNN, 線性序列, 當模型很深時，效能會下降)
   ## Exploding and Vanishing Gradients
   
   ● 在 N 層的深度網路中，必須將 N 個導數相乘才能執行梯度更新
   
   ● 導數很大，梯度會呈指數增長或“爆炸”
   
   ● 導數很小，它們就會呈指數下降或“消失”
   
> How to solve ?  --->  將前一層的輸入連接到前一層的輸出
![https://ithelp.ithome.com.tw/upload/images/20241028/20151681pWK2IUqvSB.png](https://ithelp.ithome.com.tw/upload/images/20241028/20151681pWK2IUqvSB.png)

   ● ResNet34 和 ResNet50 具有多個連續的 *3x3 卷積層* , 有不同大小特徵圖
   
   ● ResNet的輸出尺寸保持不變（padding=1, stride =1）
   
   ● ResNet 由多個 residual units 建構，
   
   ● ResNet 甚至不需要 FC Layers, 使模型更深，學習更多特徵

| **分類** | **說明**                               |
|----------|----------------------------------------|
| 優點     | 有效訓練深層網絡                       |
|          | 更好的性能                             |
|          | 模型擴展性強                           |
| 缺點     | 計算需求大（由於深層結構）             |
|          | 結構複雜（適用於行動或嵌入式設備）     |

## MobileNet
● 輕量版 CNN，能在嵌入式裝置或手機

● 推論 (Inference) 速度相對較慢 (i.e. forward propagation) 

● 低的運算能力：○ 訓練較小的模型  ○ 壓縮模型

● 使用 Pointwise Convolutions 來得到相同的輸出形狀 (對輸出進行線性組合)

● `MobileNet 適合手機的模型`：

#### **Depthwise Separable Convolutions (深度可分離卷積)**
![https://ithelp.ithome.com.tw/upload/images/20241029/20151681s7s1RSMQ7u.png](https://ithelp.ithome.com.tw/upload/images/20241029/20151681s7s1RSMQ7u.png)

#### **Two Hyper-Parameters (兩個超參數)** (兩個超參數來降低模型大小)
>
>    ● `Width Multiplier`： 縮減每一層的深度 (filters數量)
>
>    ● `Resolution Multiplier`：減少輸入影像的大小，從而減少每個後續層的大小    

| **分類** | **說明**                                |
|----------|-----------------------------------------|
| 優點     | 輕量級                                  |
|          | 高效的卷積操作                          |
|          | 靈活性：可調整 width multiplier 和 resolution multiplier |
| 缺點     | 準確度相對較低                          |

## Inception Network (解決 Filter Size 的選擇問題)
● 同時使用幾種不同大小的 Conv Filters

●  ‘same’ padding 和 stride=1 來保持尺寸大小的一致性

● 執行所有大小的 Filters，甚至是 Max Pool，然後將它們堆疊在一起, 使得模型能夠`學習高階`和`低階特徵`的組合

● `**Heavy Computation**` : Use 1x1 Convolutions to reduce the computation cost

● `**Bottleneck Layer**` : 先縮小再放大 (少了 10 倍的計算成本)

### Inception Design:  `Stem > Inception Block > Auxiliary Classifier (非每個版本都有)`

### Auxiliary Classifier（輔助分類器）: 是一種在訓練深度神經網絡（例如 Inception 網絡）時，為了改善模型訓練效果而引入的附加分類器。它通常放置在中間層，用於輔助訓練過程，並能帶來以下好處：

1. **減少梯度消失問題**：輔助分類器為中間層提供額外的損失信號，使得梯度能更有效地傳遞回前層，特別在深層網路中可以減少梯度消失的風險。

2. **加速收斂**：在訓練過程中，輔助分類器的損失會加到主分類器的損失上，這有助於模型更快地收斂。

3. **提高泛化能力**：多個分類器的損失可以起到一種正則化效果，從而提升模型在測試數據上的表現。


| **分類** | **說明**                                              |
|----------|-------------------------------------------------------|
| 優點     | 多尺度特徵提取能力                                    |
|          | 高效的計算資源利用（透過 1x1 卷積）                   |
|          | 深層網絡的可訓練性（如輔助分類器）                    |
| 缺點     | 結構較為複雜                                          |
|          | 難以移植到資源有限的設備                              |
|          | 參數和架構選擇複雜                                    |

## SqueezeNet (架構由一個獨立的 Conv Layer 和後面的 8 個 Fire Module 組成)
● 維持相同 accuracy 情況下，使用較小的 CNN 架構

     ○ 只需要少量的communication across servers
     
     ○ 使用更少的頻寬來透過雲端更快地更新模型
     
     ○ 更適合部署在嵌入式系統

● 它的`參數比 AlexNet 少 50 倍`，執行`速度快 3 倍`

● 將 3x3 `過濾器替換為 1x1` - 參數比 3x3 過濾器少 9 倍

● 將輸入到 3x3 Filters 的`頻道數減少` - 每層中參數的數量為（輸入頻道 * Filters 數 * 3 * 3）

● `晚一點`才在網路中進行 `Downsampling`，以便卷積層具有`更大的 Feature Maps`

### Fire Module - Squeeze and Expand Layers
![https://ithelp.ithome.com.tw/upload/images/20241029/20151681xDzw4MtQtD.png](https://ithelp.ithome.com.tw/upload/images/20241029/20151681xDzw4MtQtD.png)

| **分類** | **說明**                                                |
|----------|---------------------------------------------------------|
| 優點     | 極小的模型大小                                          |
|          | 相對高的分類性能                                        |
|          | 方便移植到低資源設備                                    |
| 缺點     | 性能略低於更大的網絡（相較於 VGGNet、ResNet）          |
|          | 特徵提取能力有限（例如空間特徵）                        |
|          | 相對難以擴展到更大、更深的網路（相較於 ResNet、DenseNet） |

## EfficientNet : Compound Scaling, Grid Search (網格搜尋), EfficientNet-B0 Architecture

● 透過`增加深度（層數）或寬度（Filters 數量）`來實現`縮放`

● `需要`手動調整`並不容易達到最優結果

● expand CNN Model : **Compound scaling and EfficientNet-B0** 

### Compound Scaling

● 使用一組`固定的縮放係數統一縮放每個維度（寬度、深度、解析度）`

● EfficientNet 系列模型能夠達到 **state-of-the-art accurary**，並且**效率提高 10 倍**

● 平衡所有維度的縮放的最佳整體效能

### Grid Search (網格搜尋)
● 使用網格搜尋尋找在固定資源（例如 2 倍以上的 FLOPS）下，對 `Baseline Network 進行不同維度縮放`之間的關係

● 找到每個維度最合適的`縮放係數`(可用在任何CNN), 將 Baseline Network 擴大到所需的目標模型大小

### EfficientNet-B0 Architecture
● `模型縮放`的有效性在很大程度上`取決於 Baseline Network`

● EfficientNet-B0 是使用 Google 的 (`NAS`, Neural Architecture Search) 技術設計 (平衡計算資源和性能)

   ○ 使用了 MobileNetV2 的 MBConv（Mobile Inverted Bottleneck Convolution）
   
   ○ MBConv ≈ depthwise convolution + pointwise convolution + skip connection (D+P+S)


## DenseNet : `vanishing gradients(梯度消失)`
● higher accuracy than ResNet with fewer parameters

● `Training Deep CNNs` is problematic due to `vanishing gradients(梯度消失)`

● 因為深度網路的`路徑變得很長`，`梯度`在完成路徑之前就變`為零（vanish）`

● DenseNets 透過使用 **“Collective Knowledge”** 的概念來解決這個問題，其中每一層都**接收來自所有先前層的信息** (同一個 Dense Block 裡的特徵圖大小不變)
![https://ithelp.ithome.com.tw/upload/images/20241029/20151681Q0krylSU2I.png](https://ithelp.ithome.com.tw/upload/images/20241029/20151681Q0krylSU2I.png)

● DenseNet Composition Layer 包含 Batch Norm、ReLU 和 3x3 Conv Layer
![https://ithelp.ithome.com.tw/upload/images/20241029/20151681TFNej0hbHm.png](https://ithelp.ithome.com.tw/upload/images/20241029/20151681TFNej0hbHm.png)

● Bottleneck Layer : BN-ReLU 1x1 Conv is done before BN-ReLU 3x3 Layer
![https://ithelp.ithome.com.tw/upload/images/20241029/20151681809XYpLKwc.png](https://ithelp.ithome.com.tw/upload/images/20241029/20151681809XYpLKwc.png)

● Multiple Dense Blocks with Transition Layer : 使用 1x1 Conv 和 2x2 Average Pooling 作為兩個連續密集區塊之間的 “過渡層”
![https://ithelp.ithome.com.tw/upload/images/20241029/20151681b3rYLPFNW8.png](https://ithelp.ithome.com.tw/upload/images/20241029/20151681b3rYLPFNW8.png)


## ImageNet - ILSVR : 是評估新的 CNN 模型時最常用的基準
● 優點: 

> 1. Size
>
> 2. WordNet Hierarchy
>
> 3. 幫助 AI 理解圖片裡的資訊

## Rank-N or Top-N Accuracy : 更多空間的評估分類器準確性的方法
● 考慮機率最高的前 N 個類別

