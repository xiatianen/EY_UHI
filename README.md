Urban Heat Island Machine Learning Pipeline 專案說明：

前言：
本專案源自 EY Open Science AI & Data Challenge Program，旨在建立都市熱島效應（Urban Heat Island, UHI）的機器學習預測模型，協助分析並預測熱島效應空間分布，提供未來城市規劃的決策參考。

初步結果：


專案結構：
1. preprocess.py
目的：空間資料解析度最佳化
功能：

批次多解析度重採樣 GeoTIFF（並行運算）。

結合地面觀測點（CSV），萃取不同解析度的環境變數值。

自動尋找每個變數最佳解析度（依據隨機森林特徵重要性）。

2. build_features.py
目的：特徵工程
功能：

結合地面測站與最佳解析度環境變數，建構訓練資料集。

異常值檢查與逐步特徵篩選（Stepwise）。

常見資料轉換（Box-Cox）。

探索式資料分析（EDA）。

3. train.py
目的：模型訓練與分析
功能：

訓練隨機森林回歸模型。

貝葉斯最佳化調參。

特徵重要性分析與視覺化。

輸出最終模型（.pkl），利於重複使用。