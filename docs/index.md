# 從圖片分類，走到你看得懂的 YOLO

<details class="chapter-a-toc">
<summary>本頁目錄</summary>
<ul>
<li><a href="#_1">電腦怎麼開始看懂圖片</a></li>
<li><a href="#_2">認出一張圖，還不夠應付一個場景</a></li>
<li><a href="#_3">本書沿著這些問題往前走</a></li>
<li><a href="#_4">開始前需要什麼</a></li>
</ul>
</details>

人看一張照片，很快就能認出貓、車子或行人。但對電腦來說，照片起初只是許多數字：哪一組數字代表貓？光線變了、貓轉過身，還認得出來嗎？這曾是人工智慧（AI，Artificial Intelligence）很難跨過的一道門檻。

## 電腦怎麼開始看懂圖片

早期的影像辨識，常由人先設計規則，或挑出邊緣、紋理等特徵，再交給傳統機器學習（ML，Machine Learning）方法分類。這些方法能解決一些問題，但面對種類多、背景雜、外觀變化大的照片，很難靠有限的手工設計涵蓋所有情況。

轉折需要的不只有一個更好的公式。李飛飛與研究團隊推動的 **ImageNet**，讓研究者能取得大量帶有類別標註的圖片，並在共同的任務上比較進步。到了 **2012 年**，Alex Krizhevsky、Ilya Sutskever 與 Geoffrey Hinton 團隊的 **AlexNet**，在 ImageNet 影像辨識競賽中大幅領先其他方法。大量資料、卷積神經網路與 **GPU（Graphics Processing Unit，圖形處理器）**運算在這裡交會，成為現代深度學習快速發展的重要轉折。GPU 能同時處理大量相似的乘加運算，適合加速神經網路計算。

卷積神經網路並非那年才出現。這次突破讓大家更清楚看見：與其把辨識線索逐項寫死，也可以讓多層網路從資料中學出有用的表示。後來的 **VGG** 用堆疊小卷積探索更深的網路；**ResNet** 則讓很深的網路更容易訓練。本書先學它們的核心做法，是因為偵測器也需要先從圖片讀出有用的特徵。

## 認出一張圖，還不夠應付一個場景

手寫數字辨識是一個容易想像的分類任務：一張圖給一個答案，例如「7」。照片分類也可以問「這張圖主要是什麼？」照片裡可以有背景和多個物件，但這種**整圖分類**只要求模型交出整張圖的類別。

換成路口監視器，問題就變了。同一畫面有三個行人、兩輛車，我們需要模型逐一給出每個物件的類別，以及標出範圍的 **bounding box（BBOX，包圍框）**。這叫**物件偵測（object detection）**，回答的是每個物件「是什麼」與「在哪裡」。如果畫面持續更新，還得夠快，才有時間回應正在移動的人與車；這也是自動駕駛等應用重視偵測速度的原因。

![同一張圖的兩種答案：分類只回答紅色方塊，偵測還要給位置框](assets/diagrams/classification-vs-detection.svg){ width="520" }

上圖用一個方塊把差別縮小給你看：分類回答類別，偵測還回答範圍。藍框是人畫好的示意答案；後面會讓模型學著預測它，再擴展到同圖多個物件。

**YOLO（You Only Look Once，只看一次）**在 2015 年提出，把類別與位置放在一次網路前向計算中預測，成為即時物件偵測的重要路線之一。偵測方法早在 YOLO 之前就存在；YOLO 的吸引力在於把流程整合得很直接，兼顧速度與辨識。不同版本之後仍可能需要篩選或去除重複框。

## 本書沿著這些問題往前走

我們用 PyTorch（Python 的深度學習函式庫）做很小的模型。先把一個神經元如何計算、梯度下降如何訓練弄懂，再讓 CNN 讀圖；之後才加上位置、同圖多物件與 YOLO 的不同設計。本書的教學用偵測器稱為 **MiniYOLO**。

[從 A.0：神經網路與第一次學習開始](lessons/00-warmup.md){ .md-button .md-button--primary }
[檢視完整閱讀路線](learning-path.md){ .md-button }

1. **A／第 0–3 章：模型怎麼學？**從神經元與梯度下降走到小 CNN，再檢查訓練問題、認識 ResNet 的捷徑。
2. **B／第 4–8 章：怎麼多回答位置與物件？**加入框，接起資料、訓練、推論與評估，再使用自己的圖片。
3. **C／第 9–16 章：YOLO 為什麼改設計？**用小實驗看不同版本想解決的問題、收益與代價。
4. **D／第 17–20 章：怎麼用起來？**完成結業任務，再按需求選讀影片、追蹤與部署。

全書有 52 節，包含主線 42 節與 ViT／DINO 選讀支線 10 節。[閱讀路線](learning-path.md)列出順序與支線的先備章節。每節有獨立 notebook，不必先執行上一節；閱讀則沿用已教的概念。

## 開始前需要什麼

會基本 Python，包含 `class`；數學用到高中程度的函數、指數與對數、向量與矩陣。神經網路、卷積、導數與連鎖律會在需要時用例子說明。用過 PyTorch 會比較輕鬆，也可以從本書開始學它在這些實驗中的用法。

**只讀網頁也能學。**必要解釋、手算和圖解都在本文；頁末選讀留給重做實驗或想多了解細節的人。想動手時，點頁首 Colab 按鈕，使用 Google 的線上 Python 環境。

本書的小實驗使用 CPU，資料由程式畫出。彩色幾何圖形能讓我們看清模型讀了什麼、參數是否更新、答案是否正確；真實照片的效果需要另外訓練與評估，實測範圍見[驗證範圍](status.md)。現有 notebook 固定使用 `lessons-v0.6.1` 的實驗程式。

??? note "第一次執行程式"

    Colab 需要 Google 帳號。notebook 第一格準備環境，最後一格是可修改的完整實驗；[A.0](lessons/00-warmup.md)會說明怎麼操作。網頁與 notebook 保存了實際輸出，重跑時部分訓練數字或計時可能略有不同。

??? note "故事的原始來源與教材查詢"

    - [ImageNet（2009）](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf)：資料集的建立目標與作者。
    - [AlexNet（2012）](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)：ImageNet 競賽結果、深層 CNN 與 GPU 訓練。
    - [VGG（2014）](https://arxiv.org/abs/1409.1556)、[ResNet（2015）](https://arxiv.org/abs/1512.03385)、[YOLO（2015）](https://arxiv.org/abs/1506.02640)：本文提到的設計來源。
    - [術語快速查](glossary.md)、[實驗與查證](validation/curriculum.md)、[GPU／checkpoint 實測](validation/gpu-smoke.md)。
    - [資料來源與授權](preparation/data.md)、[網站發布步驟](preparation/publish.md)、[公開課程研究](planning/course-research.md)與[讀者心得](planning/feedback.md)。
