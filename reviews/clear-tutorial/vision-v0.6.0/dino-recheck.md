# DINO 局部修正複查與評測前置補讀

本輪在 93／93 units 的首次閱讀完成、`dino-first-reader.md` 保存後進行。這是**非盲讀的 localized recheck**：協調者指出待修正卡點後，我讀當前相關正文及圖，再判斷是否已閉。原 gate raw notes、93份note快照及首次閱讀報告均未回改；不把補讀改稱原先已讀6.2或整個YOLO主線。

只讀指定的22.1、22.3、22.4相關段落／附圖，learning-path的23.2前置連結，連結所指實際 `06-evaluation.md` 全文及PR圖，再回讀23.2的評分和多物件轉接。沒有讀其他作者／技術／讀者reports、教材實作或外部來源，沒有執行訓練或數字程式。

## 複查來源與內容指紋

SHA-256 是本輪當前來源，不沿用首次閱讀凍結 packet 的指紋。

|來源|SHA-256|
|---|---|
|docs/lessons/22-views.md|`3c78cab45368317420b3287b6bac61ac3096cca2c83ace8ca4428675004ca1fc`|
|docs/assets/diagrams/22-views.svg|`c6ac553db3b63d61ac75d38b7cb98640b671434e71e1781c108815b3a24a8d31`|
|docs/lessons/22-distillation.md|`c188d3699bcbc90c91a1d215cfe53e24bd92c61bfe685964625dff21377e5ba4`|
|docs/assets/diagrams/22-distillation.svg|`e216231e715cd2a9684c3bbdc49410e3a671ac18d315d4135c031177161bc1f2`|
|docs/lessons/22-features.md|`9931f087895acdb5862d7df7b5959dec88041eb7fcbc839ef9e0104734628ed2`|
|docs/assets/diagrams/22-features.svg|`73b81c9c9853e845688921e2bfb8c532c704c4a0d84ddf021fc278b3590b547b`|
|docs/assets/diagrams/22-neighbors.svg|`951440b4c4ceeccea25d66f9bd17c4cec44ef7ba7fc16497e1d390a9b838f73a`|
|docs/learning-path.md|`52ea69edd417c873f2768fd1ed5b9a4844b7e5f3987140613f06d3cbefdceef3`|
|docs/lessons/06-evaluation.md|`8e1de6c80cd2027951169ea056feac1b02084019ff87fba6316e756c7f38206f`|
|docs/assets/diagrams/06-evaluation.svg|`8289113b070ec540923910c95227d2592d09545c0cb2e6ce215091b8ce8674fc`|
|docs/lessons/23-detection-bridge.md|`16e7dcbe497493fcfd2c2e5881194f2595ecfb2d11959ad01949334fc0f39236`|

## 原卡點是否已閉

**22-views/01 crop 約定，已閉。** 新的22-views.md在圖與第一組crop數字之前說：「圖上的 crop 座標是**原圖的像素**，順序為 `[x1,y1,x2,y2]`：左上邊界為 `(x1,y1)`，右下邊界為 `(x2,y2)`。原點在原圖左上，x 向右、y 向下；右邊 `x2` 和下邊 `y2` 不包含在裁切範圍內。」我現在能先知道數字表示邊界，再看A為28×28、B為27×27，不需要到原/04資料程式段或借框知識猜。本文與未改的22-views.svg使用上表22-views來源SHA。

**22-views/01 同seed／索引但顏色不同，已閉。** 同份新正文先說：「本節另外生成一張紅色矩形作 view 示範，不是取前章 128 張資料裡的索引 0。」圖後又說「取這份單張資料的索引 0」。現在能把前章批次圖0藍色和本頁另生成紅色辨成兩份材料，無須查生成器解釋顏色。用同一份新22-views.md SHA；原note不改。

**22-distillation/01 缺字編號，已閉於raw預覽範圍。** 新22-distillation.md用「再按 1、2、3、4追蹤更新次序」。當前圖用普通1／2／3／4；我實際重新render/view可見1舊center teacher目標、2 student backward／step、3 teacher參數EMA、4 center EMA，全部可讀，沒有方框。綠箭頭仍從本步raw teacher輸出到center，橘箭頭指teacher參數，兩者改的量清楚。新本文與圖SHA見上表22-distillation兩項。

**22-features/02 random首次意義，已閉。** 新22-features.md在「第一種讀法」開頭、鄰圖之前明說：「下文並列兩份固定的 backbone：**SSL** 是 160 步後的 teacher；**random** 保留它訓練起點、尚未更新的權重。」現在看random找到train64，能知道是在同起點未更新特徵下做最近鄰，不會猜成隨機抽參考图。後面的公平比較仍解釋同切分／規則／probe预算。用上表22-features.md SHA；鄰圖指紋未變。

**22-features/05 loss低谷不可見，已閉於raw預覽範圍。** 我重新render當前22-features.svg後，看得到約130～145步完整下探與回升，曲線不中斷、不掉出圖外；縱軸可見1.2～2.8刻度，圖源記錄上下界1.1～2.9。現可讀早期高峰、約136步低谷及後續回升，不需要推猜先前被截掉的走勢。這是呈現範圍複查，沒有重算160個原始loss或核實JSON證據。新本文／圖SHA見上表22-features兩項。

**01-small-cnn/08「表中箭頭兩邊」optional，維持原紀錄未關閉。** 它不在本輪指定修正範圍，我沒有另讀該段或補寫修正。其當時原文、來源與疑惑仍在首次閱讀note／報告。

指定的五項修正範圍內沒有剩餘需猜的原疑惑，也沒有出現新的主要概念卡點。這不等於一位新讀者重新完成盲讀驗收；我已讀過主線且知道修正位置。

## 真正讀到的6.2前置，以及回到23.2

learning-path的選讀支線段明確連「23.2 會用到 [4.1 的框與定位]及 [6.2 的 IoU]」，6.2的實際目標是 `lessons/06-evaluation.md`。我本輪才補讀該全文，不只是看路線名稱。第一次93units中已讀4.1、沒有讀6.2；此差別保留。

6.2從已有的xyxy／IoU開始，先教評估matching和訓練assignment不同。每類收集所有圖候選按score下降；每預測只和同圖同類、尚未配對GT比，IoU至少.5可作TP，一GT只配一次，否則FP，剩GT為FN。高score只是排序訊號不是正確率。主例四預測依序FP／TP／重複FP／TP，TP2 FP2 FN1；precision2/4，recall2/3，分母不同。

PR逐筆纳入的四點是(0,0)、(1/3,1/2)、(1/3,1/3)、(2/3,1/2)。我看過當前PR raw圖，能把重複框造成的下降對上同recall的兩橘點。All-points AP取右側最大precision包絡，加recall0/1占位，再求階梯矩形面積；主例前2/3高.5、其餘0，AP50=1/3，不是直線梯形或一般固定P×R。去高分背景FP變AP5/9，截score≥.85則AP1/6。每類平均是mAP；沒GT排除、GT存在沒預測為0。COCO多IoU門檻／101點及額外規則不等本教學算法；正式分數應固定官方協議。這一頁只是人工框評估無backward，我未執行程式或查官方來源。

補讀後回看23.2，我現在不只知道「54/64不是AP」這句限制，還能說理由：它只在一圖一GT／一pred下直接配對，數顏色對且IoU≥.5；沒有把所有候選依置信score排序、形成不同纳入程度的P/R點、求包絡面積，也沒有按類算AP再平均。平均IoU.6057、顏色64/64、同時門檻54/64是三個不同量，不能互相替代。

一個已教概念的變式：四張單物件圖有兩TP兩FP，固定最後P=R=.5且成功比例2/4。若score先排兩TP再兩FP，包絡0～.5高1，AP=.5；若先兩FP再兩TP，包絡0～.5高.5，AP=.25。相同固定成功比例可以有不同AP，這就是23.2沒有排序的比例不能代AP的具體原因。

回讀23.2最後的多物件轉接也順：現head只一組四數，增加第二矩形仍無第二框；NMS是已有多候選的去重，不會生框。6.2又把NMS推論去重和含GT的評估配對分開，使這個限制更明確。轉到grid需另學輸出結構／責任分配／多物資料与評分；我沒有開第7章，也不聲稱已理解其實作。沒有出現「必須先看未讀YOLO頁面才能理解本單框判准」的新斷點。

## 視覺驗證實際範圍

使用Inkscape，把當前22-views、22-distillation、22-features和06-evaluation SVG各render成寬720的PNG，再全部view。產物在 `artifacts/runs/vision-firstread/recheck-dino/`。這些是raw圖預覽，不是桌機網站截圖；不以SVG正確或PNG可讀代替Zensical檢查。

|本輪PNG|SHA-256|
|---|---|
|22-views-raw-desktop.png|`412e37f2126a129732de5fdc2a9e17870c0562a9ac37ff191c9400c2fa15a7d6`|
|22-distillation-raw-desktop.png|`63ac09a23730803147e1e87939b70671051579dde2c236d6fee327b281feac1a`|
|22-features-raw-desktop.png|`608b91db4ff16506fbe5064d077873bf7b0e9e5f0df11822a63ba9d5a019c5db`|
|06-evaluation-raw-desktop.png|`828430d9bdb2e8e062a69d7aa632259495fd0c5505818e04f39c4b896f111ddf`|

本讀者沒有操作Zensical桌機／手機或檢查其數學、頁面切換。協調者提及另一輪已查實際頁面，但我未讀那份report，沒有把別人的驗證計入自己的範圍。正文卡點的前移定義與材料識別，以及raw圖編號／曲線可讀性，在本輪指定範圍已能閉合；原先首次閱讀紀錄仍照原樣保留。
