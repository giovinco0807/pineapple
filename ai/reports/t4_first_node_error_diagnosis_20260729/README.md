# T4 First-Seat 評価器: ノード値 COMMON エラーの診断

日付: 2026-07-29
対象モデル: `D:/ofc_data/t4_first_model_pass1/evaluator_best.pt` (101-dim, test corr 0.9225 / MAE 1.156)
分析スクリプト: `ai/tutor/analyze_t4_first_common_error.py`
データ: fresh 800 roots, seed base 5,000,000(訓練 seed 770000.. / 既存分析 seed 3000000.. と非重複)、ラベルは Rust `t4_first_exact`(厳密値)
生ログ: `analysis_stdout.txt` / 数値: `analysis.json`

## 結論(1行)

**欠けている決定的事実は「相手の 2 枚同時配置を考慮した JOINT 完成分布」、とりわけ「相手のファウル確率」である。** 現行の行別独立ヒストグラムはこれを原理的に表現できず、common エラーの分散の 41% がこの 1 系統の事実で線形にすら説明される(現行特徴から再構成できる統制変数のみでは 0.9%)。コンテキストブロック(プール詳細)とジョーカーは主因ではない。

## 測定のセットアップ

- 各 root で全合法手を符号化・予測し、common エラー = mean over actions of (predicted − exact) を計算。
- 800 roots で mean|common| = **1.164**、bias −0.035、spread 0.321 — 既報の分解(common 1.079 / spread 0.309)を独立シードで再現。
- 候補となる「欠落事実」を root ごとに**厳密に**計算した:全 C(26,3)=2600 通りの相手ドローについて、捨て札 3 通り × 配置(最大 2 通り)を列挙し、相手が自己最大(royalty + FL EV、ファウル = −6)を取ると仮定した JOINT 指標。ジョーカー制約評価(`evaluate_board_with_joker_constraint`)も本番スコアラーと同一のものを使用。

## 発見 1(主因): 相手の JOINT ファウル確率・JOINT 到達 EV が特徴に存在しない

### 相関(signed common error, n=800)

| 候補事実 | r | 統制(11 変数)への追加 ΔR² |
|---|---|---|
| **joint_foul_rate**(相手が全配置でファウルする確率) | **−0.470** | **+0.386** |
| **joint_ev**(相手ベスト完成の自己値の期待値) | **+0.453** | **+0.383** |
| joint_royalty | +0.183 | +0.063 |
| joint_std | +0.128 | +0.018 |
| joint_fl | +0.080 | +0.009 |
| (統制のみ: 現行特徴から作れる indep_royalty / indep_fl / locked_cat / joker 数 / cat_slack / pool_high / hero 側平均) | 各 \|r\| ≤ 0.07 | R² = **0.009** |
| 統制 + JOINT 全部 | — | R² = **0.415** |

単独でも foul_rate の R² = 0.221、joint_ev = 0.205。統制変数だけでは何も説明できない(R² 0.009)ことが、「モデルが現行特徴では原理的に知り得ない事実」であることの証拠になっている。

### ファウル確率で層別した bias(決定的なパターン)

| joint foul rate | n | mean\|common\| | bias |
|---|---|---|---|
| [0.000, 0.001) | 129 | 1.075 | **+0.742** |
| [0.001, 0.050) | 135 | 1.131 | **+0.935** |
| [0.050, 0.250) | 104 | 0.954 | +0.516 |
| [0.250, 0.750) | 186 | 1.122 | −0.118 |
| [0.750, 1.010) | 246 | 1.351 | **−1.146** |

モデルはファウル情報を持たないため「平均的な相手」を仮定する: 相手が確実に生き残る root では楽観しすぎ(+0.9)、相手がファウル確定の root では悲観しすぎ(−1.1)。**2 点幅の系統的バイアス**であり、これが common エラーの正負両テールを作っている。

### 実ハンド(負テール: 相手ファウル確定を見えていない)

800 root 中 199 root(25%)で foul_rate = 1.000(相手ファウル確定)。うち現行の category 比較 `locked_cat_any` が検出できたのは **75/199 (38%)**。以下は未検出の典型 3 型。

**(a) 同カテゴリ内の value 差 — seed 5000103, common −6.28**
```
BB (hero)  top: Qd Qs Kh [pair]   mid: Ts 3s 9h [3/5]   bot: X2 8s 6c 5d 6h [trips]
BTN (opp)  top: 7s [1/3]          mid: 4s Ad Kd Ah Th [pair A]   bot: 7c 4c 8h 3c X1 [pair 8]
exact: 全手 0.000 (hero 必然バスト & opp ファウル確定 → 0-0)   pred: −6.10 〜 −6.37
```
相手 mid はエースのペアで完成、bot はジョーカー込みでも最大 8 のペア。**pair > pair でカテゴリが同じ**ため現行の `cats[1] > cats[2]` 比較は沈黙する。value(kicker まで含む encoded hand value)で比較すれば 1 比較で判る。

**(b) 未完成行の到達上限 — seed 5000568, common −6.01**
```
BTN (opp)  top: Kc Ks [2/3, pair K 確定]   mid: 4d 8c Ts Js [4/5]   bot: straight 完成
```
mid は残り 1 枚で最高でも pair J。top は既に pair K 以上が確定。**「上の行の到達下限 > 下の行の到達上限」**という未完成行を跨いだ事実で、完成行同士の比較では原理的に検出不能。JOINT 列挙なら自動的に foul_rate = 1.0 と出る。

**(c) kicker が決める lock — seed 5000763, common −7.28(最悪)**
```
BTN (opp)  top: 5d As [2/3]   mid: 5s Kd 7h Qc 4d [K-high 完成]   bot: two-pair
exact: 全手 0.000   pred: −6.86 〜 −8.07 (node_err −6.86)
```
mid が K-high で完成しており、top には既に As がある → top のどの完成も A-high 以上 > K-high でファウル確定。カテゴリは high vs high で同じ。regular track の header にある「a kicker that decided a line」の相手側版。

なお locked_cat が立った root でも(例 seed 5000494, locked_cat=1, common −6.05)誤差は −6 のまま残る。101 次元中の 1 バイナリでは、訓練データで稀な「相手ファウル確定 → hero バストでも 0 点」という ±6〜12 点の跳びを担いきれていない。**連続値の foul_rate が必要**。

### 実ハンド(正テール: 生き残る強い相手を見えていない)

**seed 5000034, common +5.33** — hero は全手でバスト確定(mid pair 9 > bot pair 3)。相手 foul 0.59。exact −4.89 に対し **pred +0.43〜+0.46**。hero バスト確定なら exact ≤ 0 が恒等的に成り立つのに、モデルは正値を出す。hero バスト時の結果は純粋に「相手が生き残るか・相手の合計点」で決まるため、欠落事実の影響が最大化される(hero 全バスト root は n=346, mean|common| 1.254)。

**seed 5000025, common +8.17(正側最悪)** — 相手 mid KKs+Kc(トリプル K ドロー)、bot pair A、foul 0.000、joint_ev +6.1。exact −12.1 の手をモデルは −6.0 と評価。相手の「到達スコア分布」の要約(joint_ev, joint_std=8.6)が無いため強い相手を割り引けない。

**seed 5000706, common +6.06** — 相手 top が pair A 完成で行別 FL 期待値は 29.9 と符号化されるが、mid pair A vs bot(要 2 枚)で foul 0.835、**FL は survive 条件付きでしか実現しない**ため joint_fl は 4.9。行別 FL ヒストグラムがファウルと独立に足し込まれる構造欠陥の実例(FL チェーン 0〜63.5 点のスケールでは致命的)。

## 発見 2: JOINT 事実だけで組んだ無学習の近似式がテールをほぼ解消する

foul_rate / joint_royalty / joint_fl / hero 側の bust・royalty・FL と行別勝敗期待から機械的に合成した `approx_ev`(学習なし、スクリプト参照)は:

| 指標 | 学習済みモデル (101-dim) | approx_ev(式のみ) |
|---|---|---|
| corr vs exact_mean (800 roots) | 0.914 | 0.813 |
| node MAE(全体) | 1.164 | **1.004** |
| node MAE(モデル最悪 75 roots, \|common\|>3) | 4.482 | **0.778** |

つまり JOINT 事実は「あれば良い」程度ではなく、**それ単独でテール(最大 10 点級の暴発)をほぼ消せる**情報量を持つ。バルク(|common|≤3 の 725 root, モデル MAE 0.821)でも foul の相関 −0.380 / joint_ev +0.343 が残っており、再学習時はバルクにも効くと期待できる(線形補正では効かないが、それは線形プローブの限界であり、式 approx_ev のバルク MAE ~1.0 が非線形での有用性を示している)。

## 仮説の判定

| 仮説 | 判定 | 根拠 |
|---|---|---|
| 1. 行別独立ヒストグラムは同時達成を過大評価し、相手の選択を無視 | **支持(ただし主戦場はファウル)** | 帰結のうち支配的なのは JOINT ファウル事実(r=−0.470, ΔR²+0.386)。royalty の過大評価そのもの(overstate_royfl)は r=−0.044 で直接の線形ドライバではない。joint_royalty ΔR²+0.063、FL の過大評価は FL ハンドで大きい(seed 5000706: 29.9 vs 4.9) |
| 2. 相手の到達最終強度/スコア分布の要約が無い | **支持** | joint_ev r=+0.453, ΔR²+0.383。分散(joint_std)は小さい寄与(+0.018)。分布そのものより期待値+ファウル率でほぼ足りる |
| 3. コンテキストブロックにプールのランク/スート詳細が無い | **反証** | pool_high_frac r=−0.007、プール系統制はすべて \|r\|<0.05。プール詳細が効くのは相手の行完成を通じてであり、それは行別ヒストグラム(厳密)が既に担っている。hero 盤面は完成済みなのでプール詳細の独立効果は無い |
| 4. ジョーカー(相手 top / プール内)が FL チェーンを通じて common エラーを駆動 | **反証(主因としては)** | mean\|common\| は pool_jokers=0/1/2 で 1.197/1.125/1.223、opp_jokers=0/1/2 で 1.156/1.189/1.094 とほぼ平坦。bias も ±0.1〜0.2 に留まり、foul 効果(±1.1)の 1/10 のオーダー |

追加で検証した仮説:
- **行別 head-to-head 勝敗期待(exp_lines)**: r=−0.016, ΔR²+0.000 で**反証**。ライン勝敗の情報は hero ブロック + joint ブロックの符号比較で既に足りている。regular track の OUTCOME ブロック相当の欠落は今回の common エラーの原因ではない。
- **hero バスト時の上限違反**: モデルは hero 全バスト root で正の EV を出すことがある(seed 5000034)。これは新特徴ではなく、foul_rate が入れば「バスト時 EV = −6(1−foul) − E[相手合計]」という構造を網が学べるようになる、という形で解消されるべきもの。

## 提案(優先順位つき)

### P1: 相手 JOINT 完成ブロックの追加(opponent block 41 → 49, +8 dims)【最優先】

全 C(26,3) ドロー × 捨て札 3 × 配置 ≤2 の厳密列挙(相手は自己最大 = royalty + FL EV、ファウル=−6 を選択)から:

1. `joint_foul_rate` — 全配置ファウルとなるドローの割合(1 dim)
2. `joint_ev` — ベスト完成の自己値期待 / 正規化(1 dim)
3. `joint_royalty_surv`, `joint_fl_surv` — survive 条件付き royalty / FL 期待(2 dims)。FL をファウルと独立に足す現行の構造欠陥をここで直す
4. `joint_std` — ベスト値の標準偏差(1 dim)
5. `p_best_ge_6`, `p_best_ge_15` — 到達スコアのテール確率(2 dims)
6. `joint_ev_surv` — survive 条件付き期待(1 dim)

計算コスト(実測): 純 Python 実装で **~0.17 s/root**(行別完成テーブル 26+325 評価を前計算すれば、列挙本体は 2600×6 のルックアップ+比較)。Rust 移植なら **~1–2 ms/node** の見積もりで、これはノードあたり 1 回のみ(action 非依存)。T3 探索のリーフ用途でも、regular track の `JOINT_DRAWS = 512` と同じ層化ストライドで 1/5 に落とせる(foul_rate の SE は 512 サンプルで ~2%)。厳密解(~26 ms/root)よりは 1 桁以上安い。

期待効果: 線形プローブで common 分散の 41%、無学習の合成式でノード MAE 1.164→1.004・最悪 75 root で 4.48→0.78。再学習後は「1.0 pt 以内 64.8%」の大幅改善と最悪誤差 10 pt 級の解消を見込む(保証ではない。特に bulk ±0.8 の残差は本分析のどの候補とも強くは相関せず、モデル容量/ラベル分布側の可能性がある)。

### P2: joint ブロックの locked-foul 検出を category 比較から value 比較へ(+0〜3 dims)【P1 が入るなら任意】

現行 `cats[1] > cats[2]` を encoded hand value の比較に変更し、さらに「未完成上行の到達下限 vs 完成/未完成下行の到達上限」(行別完成テーブルの min/max から O(1))を追加。検出率 38% → ほぼ 100%(ジョーカー制約ケースのみ列挙が必要)。P1 の foul_rate はこれを連続値で包含するため、P1 実装後は解釈可能性/安全網としての価値のみ。

### P3: コンテキストブロックは現状維持【変更不要】

プール詳細の追加は測定上正当化されない(発見 3)。次元を増やすならその分を P1 に使うべき。

## 再現

```
python -m ai.tutor.analyze_t4_first_common_error --roots 800 --show 12 \
    --out ai/reports/t4_first_node_error_diagnosis_20260729/analysis.json
```
(Rust ソルバのラベル付け ~5 s、JOINT 列挙込み全体 ~140 s)
