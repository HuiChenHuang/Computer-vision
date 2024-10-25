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
