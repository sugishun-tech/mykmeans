# Cythonでk-meansを一から実装：SOTA調査と実測

`mykmeans` 0.2.0を実装した。目的は、`sklearn.cluster.KMeans`に近いインターフェイスで、CPU上の計算速度と数学的な解析機能を両立することにある。密行列にはLloyd、Hamerly、Elkan、Exponion、CSR疎行列にはLloydとElkanを用意した。

先に結論を書くと、今回の測定では低次元の一様分布でHamerlyとExponionが有効だった。一方、重なりの強いデータやCSRではscikit-learnより遅い条件がある。世界最速やSoTA更新を実証したものではない。高速化手法を事前に調べ、独立に実装し、解の品質を含めて比較した記録だ。

## 「一から実装」の範囲

学習、k-means++初期化、予測、評価指標、入力検証、estimatorのパラメータ管理を、このリポジトリ内に実装した。scikit-learnのクラスタリング処理やprivate関数へ委譲する経路はない。外部ライブラリの実装コードも転載していない。

ただし、NumPyの配列演算、SciPyのBLAS、汎用距離計算、疎行列、固有値分解、線形割当などは利用する。「独立したクラスタリング実装」と「数値計算部品まで無依存」は区別した。実行時の必須依存はNumPy、SciPy、threadpoolctl。Cythonはビルド時に使用する。

scikit-learnはPipelineやGridSearchCVとの連携、および比較テスト用の任意依存になる。連携時にはタグやmetadata requestのオブジェクトを使うが、数値計算は移譲しない。[1]

## 導入と基本操作

Python 3.10以降とCコンパイラが必要。配布zipを展開した後、ソースからビルドする。GitHubの所有者URLやPyPIへの公開済みパッケージは前提にしない。

```bash
cd mykmeans
python -m pip install .
```

LinuxではGCC、WindowsではVisual Studio Build Toolsを利用する。通常ビルドはCPU固有の命令セットを強制しない。手元のCPUへ最適化する場合は、対応コンパイラで`MYKMEANS_NATIVE=1`を指定できる。OpenMPを無効にする場合は`MYKMEANS_OPENMP=0`を使う。以下はLinuxでの指定例。

```bash
CC=gcc MYKMEANS_NATIVE=1 python -m pip install .
```

`-march=native`付きのバイナリを異なるCPUへそのまま配布することは想定しない。本記事のベンチマークは、この追加最適化を使わない通常ビルドで測った。

```python
import numpy as np
from mykmeans import KMeans

rng = np.random.default_rng(0)
base = rng.normal(size=(16, 8)) * 5
X = base[rng.integers(16, size=20_000)] + rng.normal(size=(20_000, 8))
weights = np.linspace(0.5, 2.0, len(X))

km = KMeans(
    16, algorithm="exponion", n_init="auto",
    random_state=0, n_threads=4, record_history=True,
).fit(X, sample_weight=weights)

print(km.inertia_, km.n_iter_, km.backend_)
labels = km.predict(X)
distance = km.transform(X[:10])
squared = km.transform_squared(X[:10])
nearest_squared = km.predict_distances(X[:10])
```

`transform`はユークリッド距離を返す。追加の二乗距離メソッドはfloat64出力。標準引数と既定値は公開KMeans APIに合わせ、`n_threads`と`record_history`を追加した。`init`はk-means++、ランダム、中心配列、callableに対応する。[1]

## 実装前に調べた高速化研究

調査日は2026-11-17。今回追加したElkan、Exponion、CSRの実装に着手する前に一次資料を確認した。速度の比較では、CPUかGPUか、厳密な割当か近似か、初期化を含むかを分ける必要がある。

| 方式 | 高速化の考え方 | 本実装での扱い |
| --- | --- | --- |
| Elkan（2003）[2] | 点・中心ごとの下界と中心間距離 | dense / CSRに実装 |
| Hamerly（2010）[3] | 点ごとの少数の上下界 | denseに実装 |
| Yinyang（2015）[4] | 中心をグループ化した下界 | 調査済み、未実装 |
| Exponion（2016）[5] | 最近傍候補を幾何学的な球へ限定 | 全近傍ソートを使う変種を実装 |
| Geometric-k-means（2026）[6] | スカラー射影による対象点の絞り込み | 調査済み、未実装 |
| FastKPP（2025–2026）[7] | 三角不等式・ノルムによる初期化の省略 | 三角不等式フィルタだけ独立に導出して採用 |
| SuperKMeans（2026）[8] | 部分GEMMと次元剪定、検索recallによる停止 | 調査済み、未実装 |
| Flash-KMeans（2026）[9] | GPU上で距離とargminを融合 | 調査済み、未実装 |

SuperKMeansは高次元埋め込み検索を対象に、ランダム回転とADSamplingの仮説検定を使う。「検索品質の低下がほぼない」という結果は、通常のLloydとすべての割当・停止条件が同じという主張ではない。Flash-KMeansはGPUのメモリ転送とカーネル融合を中心に設計されている。[8][9]

これらの論文の最大速度倍率を、今回のCython実装の実測値としては扱わない。比較対象をscikit-learnに置いた今回の実験だけでは、最新実装全体に対する順位も分からない。

## 目的関数と中心更新

観測$x_i\in\mathbb{R}^d$、中心$c_j$、非負重み$w_i$に対し、最小化する量は

$$
J(C)=\sum_{i=1}^{n}w_i\min_{1\le j\le k}\|x_i-c_j\|^2
$$

となる。重みの合計は正とする。所属$z_i$を固定し、$m_j=\sum_{i:z_i=j}w_i>0$なら、中心更新は

$$
c_j=\frac{1}{m_j}\sum_{i:z_i=j}w_i x_i.
$$

ゼロ重みの観測は許すが、負・非有限・合計ゼロの重みは拒否する。空クラスタでは、重み付き残差の大きい正重み点を、元のクラスタに正の重みを残せる場合に移す。異なる点が不足する入力では重複中心を許し、観測されたクラスタ数が要求数より少ないことを警告する。

## Cythonカーネルの構成

### Lloyd、BLAS、並列集計

`src/mykmeans/_core.pyx`がdense、`_sparse.pyx`がCSR、`_distance.h`が直接距離などの小さなCカーネルを担当する。Python側の`_kmeans.py`は初期化・複数開始・属性の管理、`_compat.py`は独立したestimator基盤を受け持つ。

点・中心・座標の内側のループをCythonの型付きコードに置き、GILを解放してOpenMPで分割する。workerごとの座標和と重みを最後に合算する。float32入力でも、重心の集計と直接距離の累積はfloat64で行う。保存する中心のdtypeは入力に合わせる。[11]

十分な次元とクラスタ数があるLloydでは、

$$
\|x-c\|^2=\|x\|^2+\|c\|^2-2x^\mathsf{T}c
$$

をブロックGEMMへ接続する。全点・全中心の距離行列は保存しない。ただしノルム展開には桁落ちがあるため、誤差の余裕を使って候補を選別し、選ばれた中心や曖昧な候補は差ベクトルから直接評価する。

BLASの内部並列を抑制し、OpenMP側で並列度を管理する。小さい問題は並列化の準備費用が目立つためシリアルにする。`n_threads=4`は常に4倍速という指定ではない。

### Hamerly、Elkan、Exponion

Hamerlyは、所属中心への上界$u_i$、競合中心への下界$l_i$、中心間距離の半分の最小値

$$
s_j=\tfrac12\min_{h\ne j}\|c_j-c_h\|
$$

を持つ。$u_i<\max(l_i,s_{z_i})$なら、所属が変わらないと判断できる。中心移動量に応じて境界を更新する。所属が変わった点だけ集計を増減し、定期的に全再集計して誤差の蓄積を抑える。[3]

Elkanは各点・各中心への下界を保持するため、より細かく省略できる。ただしfloat64の下界だけで`8*n*k` bytesが必要になる。距離評価数が少なくても、境界更新とメモリ転送で遅くなる場合がある。[2]

ExponionはHamerly型の境界に候補球を加える。現在の所属中心を$a$、その中心から別の中心までの最短距離を$s_a$とすると、点の最近傍・第2近傍中心は、$a$を中心とする半径$2u_i+s_a$の球内で探せる。球外の中心への再評価を省く。[5]

今回は中心ごとの全近傍ソートを使った。原論文の部分ソート方式とは異なり、中心間領域は`O(k²)`、ソートは反復あたり`O(k² log k)`になる。アルゴリズム名だけで原論文と同じ性能だとは考えない。

これらの省略規則は実数の厳密演算では安全だが、実装は浮動小数点で動く。境界に保守的な余裕を設け、曖昧なら直接計算する。区間演算による完全な誤差証明や、異なる実装との全ビット一致を保証したものではない。

### k-means++も初期化から計測する

基礎となるk-means++はD² samplingで初期中心を選ぶ。[10] 本実装では$w_iD_i^2$に比例して候補を抽出し、複数のlocal trialsからpotentialの小さい候補を採用するweighted greedy方式を使う。標準のtrial数は`2 + int(log(k))`。

既存の最近傍中心を$a$、新候補を$c$、$r=\|x-a\|$とすると、

$$
\|a-c\|>2r
\implies\|x-c\|\ge\|a-c\|-r>r.
$$

したがって、その点の最小距離を更新しなくてよい。このフィルタを独立に実装した。FastKPP全体の移植ではなく、ノルムフィルタは含めていない。[7]

フィルタにも費用がかかるため、`d>=8, n>=1024, k>=8`の場合に、最大256点で省略率を見積もる。64次元以上では60%、それ未満では75%を目安に経路を選ぶ。見積もりは実行経路だけを変え、sampling potentialの計算対象を減らしたり、greedy trialsを近似したりしない。

### CSRは独立した疎行列カーネル

CSRはLloydとElkanを実装した。`algorithm="auto"`はCSR Lloyd、HamerlyとExponionの明示指定は`ValueError`になる。

入力全体を密行列へ変換しないが、中心は密な`k × d`配列で、worker別集計にも`O(Tkd)`が必要になる。CSR Elkanにはさらに`n × k`下界がある。「入力が疎なら全workspaceも小さい」とは限らない。

## 公平な条件で測る

測定環境はAMD EPYC 9V74、利用可能な論理CPU 9、Linux x86_64、Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0、scikit-learn 1.9.1、Cython 3.3.1、GCC 14.2.0。通常ビルドの`-O3 -fopenmp`を使い、`fast-math`は無効。

全実装で`n_init=1`、`max_iter=100`、`tol=1e-4`を指定した。ウォームアップ1回の後に`fit()`全体を3回測り、中央値を比較する。データ生成、ウォームアップ、独立した目的関数の再計算、ARI計算は計測から除く。

比較を二つに分けた。

1. **固定初期中心**：すべての実装へ同じ中心配列を渡す。
2. **k-means++込み**：各実装が初期化する。同じseedでも同じ初期中心になる保証はない。

外側のスレッド上限を揃え、mykmeansにも同じ`n_threads`を渡す。共有コンテナ上の測定なので、3回の中央値だけで環境差や時間変動を取り除けたとは考えない。

### dense：1スレッド・float64・固定初期中心

単位はms。左端の数値は標本数、次元数、クラスタ数を表す。

| データ、n / d / k | sklearn Lloyd | my Lloyd | my Hamerly | my Elkan | my Exponion |
| --- | ---: | ---: | ---: | ---: | ---: |
| 分離、25,000 / 3 / 32 | 10.723 | 9.398 | 9.923 | 20.741 | 9.148 |
| 重なり、12,000 / 16 / 16 | 8.965 | 13.908 | 14.518 | 22.155 | 15.542 |
| 高次元、5,000 / 128 / 32 | 15.146 | 14.871 | 20.944 | 40.512 | 21.517 |
| 一様、30,000 / 2 / 64 | 124.955 | 114.039 | 60.058 | 400.120 | 43.120 |

一様2Dでは全実装が63反復で、Exponionはsklearn Lloyd比で約2.90倍。反対に、重なり16Dでは自作の全方式が遅く、Elkanは一様2Dでも400.120msかかった。距離の省略数だけでは実時間を説明できない。

4スレッドの同じ一様2D・float64・固定初期中心では、sklearn Lloydが58.067ms、Hamerlyが19.644ms、Exponionが21.583msだった。倍率はそれぞれ約2.956倍、2.690倍。1スレッドと4スレッドでHamerlyとExponionの順位も入れ替わった。

float64の固定初期中心では、両スレッド設定の全ケースでsklearn Lloydに対するARIが1.0だった。一方、float32の固定初期中心・一様2DではARIが約0.998189、独立評価したSSEの差が約+0.003073%のケースがある。この差を完全一致とは表現しない。

mykmeansの報告inertiaと公開中心から直接計算したinertiaの最大相対差は約`3.32e-14`だった。これは報告値の整合性の確認であり、scikit-learnと同じ局所解になるという意味ではない。初期化込みの大きな倍率も、反復数と解の品質を併せて読む必要がある。

### CSR：速い条件だけを掲載しない

次はfloat64、1スレッド、固定初期中心。疎の位置を一様に生成し、格納値を標準正規分布から作ったデータで、強いクラスタ構造を与えていない。

| n / d / k、密度 | sklearn Lloyd | my CSR Lloyd | my CSR Elkan |
| --- | ---: | ---: | ---: |
| 12,000 / 1,000 / 16、1% | 19.265 | 32.124 | 25.948 |
| 20,000 / 64 / 16、10% | 76.379 | 145.163 | 120.763 |

初期化込みを含むCSRの全比較で、`sklearn Lloyd時間 / mykmeans時間`は約0.526〜0.816倍。今回はすべてsklearn Lloydより遅かった。ARIと反復数は比較先と一致しており、独立実装として動作することと、速度で勝つことは別の検証になる。

### 生データと再現

denseは各スレッド設定で4データ×2dtype×2初期化×6実装の96レコード。6実装にはsklearn Elkanも含む。CSRは16レコードで、合計208レコードを次の3ファイルに保存した。

- `benchmarks/results_portable_sklearn191_1thread.json`
- `benchmarks/results_portable_sklearn191_4threads.json`
- `benchmarks/results_csr_sklearn191_1thread.json`

各回の時間、最小・最大・中央値、反復数、目的関数、ARI、バージョン、スレッド設定、ソースのSHA-256を記録した。表を再計算できるようにし、都合のよい最大倍率だけを残さない。

```bash
python benchmarks/benchmark.py --quick --threads 1 --repeats 3 --dtypes float32,float64 --output benchmarks/results.json
python benchmarks/benchmark.py --quick --threads 4 --repeats 3 --dtypes float32,float64 --output benchmarks/results_4threads.json
python benchmarks/benchmark_sparse.py --threads 1 --repeats 3 --output benchmarks/results_csr.json
```

## 数学的な解析API

解析は明示的な呼出しに分け、通常の`fit()`で共分散やsilhouetteを計算しない。

### 重み付きANOVAと早期停止

クラスタの重み付き平均を$\mu_j$、全体平均を$\mu$とすると、

$$
\underbrace{\sum_iw_i\|x_i-\mu\|^2}_{\mathrm{total\_ss}}
=\underbrace{\sum_iw_i\|x_i-\mu_{z_i}\|^2}_{\mathrm{within\_ss}}
+\underbrace{\sum_jm_j\|\mu_j-\mu\|^2}_{\mathrm{between\_ss}}.
$$

しかし、最終割当後の平均と保存中心$c_j$は、早期停止などで一致しない場合がある。そのとき

$$
J(C,z)=\mathrm{within\_ss}
+\underbrace{\sum_jm_j\|c_j-\mu_j\|^2}_{\mathrm{center\_offset\_ss}}.
$$

後者を別に返すことで、inertiaを無条件に平均まわりの分散と同一視しない。平均は正のクラスタ質量について定義し、ゼロ質量クラスタは和から除く。

```python
stats = km.diagnostics(X, sample_weight=weights, covariance="diag")
print(stats["total_ss"], stats["within_ss"], stats["between_ss"])
print(stats["inertia"], stats["center_offset_ss"])
```

`covariance="full"`は完全な共分散、`"none"`は共分散なし。共分散は質量$m_j$で割る母共分散で、あらゆる重みの解釈に対する不偏推定量ではない。クラスタ別SSE、半径、重み、標本数も取得できる。

### 固定割当の勾配とHessian

所属を固定した二次式について、

$$
\nabla_{c_j}J=2m_j(c_j-\mu_j),\qquad
\nabla^2J=\operatorname{blockdiag}(2m_1I_d,\ldots,2m_kI_d).
$$

```python
from mykmeans import objective_gradient, assignment_hessian

gradient = objective_gradient(
    X, km.cluster_centers_, labels=km.labels_, sample_weight=weights,
)
hessian = assignment_hessian(
    km.labels_, km.n_clusters, X.shape[1], sample_weight=weights,
)
```

Hessianの既定出力はCSR。最近傍中心が切り替わる境界では目的関数が微分可能とは限らないため、この式は固定割当の枝に対する微分として扱う。

### PSD計量、Voronoi領域、soft assignment

正半定値行列を$M=AA^\mathsf{T}$と分解すると、行ベクトルを$xA$へ変換して

$$
\|(x-y)A\|^2=(x-y)M(x-y)^\mathsf{T}
$$

をEuclidean k-meansで扱える。

```python
from mykmeans import FeatureMetric, voronoi_halfspaces

M = np.diag([1., 4., 0.5, 1., 1., 1., 1., 0.])
metric = FeatureMetric(metric=M)
Z = metric.fit_transform(X)
A, b = voronoi_halfspaces(km.cluster_centers_, index=0)
q = km.soft_assign(X[:10], temperature=2.0)
```

`FeatureMetric(feature_weights=...)`も指定できる。$M$が特異なら擬距離であり、疑似逆行列による逆変換は零空間の情報を復元しない。標準化や計量学習を暗黙に行う機能でもない。

`A @ x <= b`は中心0の閉じたVoronoiセルを表す。`power_weights`で加法的な重みを持つpower diagramにもできる。これは観測の`sample_weight`とは別物だ。

soft assignmentは$T>0$に対して

$$
q_{ij}=\frac{\exp(-\|x_i-c_j\|^2/T)}{\sum_h\exp(-\|x_i-c_h\|^2/T)}
$$

を返す。行ごとに和が1になるGibbs重みだが、校正済み確率や混合モデルの事後確率を保証しない。`transform`と同様、出力の`n × k`メモリが必要になる。

### 評価指標とクラスタ数

`adjusted_rand_score`、`silhouette_samples`、`silhouette_score`、`calinski_harabasz_score`、`davies_bouldin_score`も独立実装した。これらは非重み付きで、学習時の重みを自動的には適用しない。Silhouetteの距離workspaceは分割するが、全点評価の演算量は`O(n²d)`のままなので、必要に応じて標本数を制限する。

```python
from mykmeans import evaluate_k, clustering_stability, silhouette_score

quality = silhouette_score(X, km.labels_, sample_size=2000, random_state=0)
scores = evaluate_k(X, range(2, 7), n_init=3, n_threads=1)
stability = clustering_stability(X, 16, seeds=[0, 1, 2], n_init=1, n_threads=1)
```

安定性は同じデータで初期値を変えたARIであり、bootstrap安定性ではない。`align_centers`と`align_labels`では中心やラベルの対応づけもできる。`gap_statistic`は座標軸またはPCA座標の一様参照分布を使い、候補数$m$、参照数$B$に対して$m(B+1)$回の学習を行う。非重み付きの明示的な解析で、唯一の正しい$k$を保証しない。[12]

## 数値的一貫性と互換性の限界

密行列は中心化して計算する。ただし元座標に中心を戻す際に丸めが起こり、境界上の点が別の中心へ移ることがある。すべての開始で、実際に保存する中心に対して元座標で最終割当と目的関数を評価し、その値で最良の開始を選ぶ。`copy_x=False`は入力を一時変更して復元するため、減算・加算の丸めは残り得る。

`get_params`、`set_params`、clone用フック、特徴量名、出力形式は独自基盤に実装した。Pipeline等と連携する場合は`python -m pip install ".[sklearn]"`を使う。pandas出力はpandas、polars出力はpolarsを別途必要とする。

互換性はAPIと振る舞いの対応範囲を意味し、全ビット一致や全estimator check通過の主張ではない。クラスタ番号は任意なので、比較にはARIや中心の対応づけも必要になる。

今回のsklearn 1.9.1の汎用estimator checksでは、KMeansが52成功、2不一致、1skip。2件は整数重みと明示的な行複製の比較で、中心番号の順序が異なる検査だった。この2件はARIが1.0、目的関数が一致し、中心番号を揃えると予測も一致した。ただし、他のデータで独立初期化が別の局所解へ進まないという一般保証にはならない。FeatureMetricは46成功、0不一致、1skipだった。

公開用pytestはscikit-learn 1.6.1、1.9.1の両環境で**259 passed**。sklearn namespaceのimportを禁止した独立動作、dense / CSR、重み、空クラスタ、境界、readonly入力、勾配の有限差分、geometry、metrics、Pipelineを検証した。OpenMPなしの再ビルドでも259件が通過し、sklearnを導入していない新しい仮想環境へ作成したwheelをインストールした数値APIのスモークテストも成功した。CI定義は同梱するが、その存在をWindows / macOSでの実行実績とは数えない。

```bash
python -m pip install ".[dev]"
python -m pytest -q
python examples/basic.py
python examples/research.py
```

対象はCPU上のバッチ型Euclidean k-means。GPU、分散、`partial_fit`、球面k-means、制約付きクラスタリング、大域最適解の保証は含まない。ソース、テスト、数式の説明、計測コード、生データ、MITライセンスを`mykmeans`にまとめた。次の最適化は、今回遅かったCSRや高次元の条件を、同じ品質基準で改善できるかを基準に評価する。

## 参考文献

1. [scikit-learn: KMeans API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
2. [Elkan: Using the Triangle Inequality to Accelerate k-Means, ICML 2003](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf)
3. [Hamerly: Making k-means even faster, SDM 2010](https://cs.baylor.edu/~hamerly/papers/sdm_2010.pdf)
4. [Ding et al.: Yinyang K-Means, ICML 2015](https://proceedings.mlr.press/v37/ding15.html)
5. [Newling & Fleuret: Fast k-Means with Accurate Bounds, ICML 2016](https://proceedings.mlr.press/v48/newling16.html)
6. [Geometric-k-means, Machine Learning, 2026](https://link.springer.com/article/10.1007/s10994-025-06891-1)
7. [FastKPP: 著者による実装・2025/2026年の論文案内](https://github.com/GuilleRodri/FastKmeansPP)
8. [Kuffo et al.: A Super Fast K-means for Indexing Vector Embeddings, 2026](https://arxiv.org/html/2603.20009v1)
9. [Flash-KMeans, 2026](https://arxiv.org/abs/2603.09229)
10. [Arthur & Vassilvitskii: k-means++ — The Advantages of Careful Seeding](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
11. [Cython: Using Parallelism](https://cython.readthedocs.io/en/stable/src/userguide/parallelism.html)
12. [Tibshirani, Walther & Hastie: Estimating the number of clusters in a data set via the gap statistic](https://doi.org/10.1111/1467-9868.00293)
