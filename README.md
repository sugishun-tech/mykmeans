# mykmeans

Cython / OpenMPで独立に実装した、重み付きCPU k-means。`sklearn.cluster.KMeans`の主要インターフェイスに合わせ、数学的な解析機能を追加したライブラリ。

**0.2.0では、密行列・CSR疎行列の学習、初期化、予測、評価指標をすべて独立実装に統一した。** scikit-learnのクラスタリング・初期化・評価・検証処理は呼び出さない。実行時の必須依存はNumPy、SciPy、threadpoolctlだけ。NumPyの配列演算やSciPyのBLAS・距離計算など、汎用的な数値計算部品は利用する。

密行列はLloyd、Hamerly、Elkan、Exponionの4方式、CSR疎行列はLloydとElkanに対応する。速度は次元、クラスタ数、分離の程度、初期化、CPU、BLASによって変わる。SoTAの調査と採用判断は[高速化手法の調査](docs/sota.md)、再現条件・生の測定値・解の品質は[ベンチマーク](benchmarks/README.md)を参照。

## インストール

Python 3.10以降とCコンパイラが必要。ビルド時にCython、NumPy、SciPyなどを使用する。以下は同梱ソースからのインストールで、PyPIへの公開済みパッケージや配布wheelを前提にしない。

```bash
cd mykmeans
python -m pip install .
```

Debian / Ubuntuでの準備と、実行するCPU向けの最適化ビルド:

```bash
sudo apt install build-essential python3-dev python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
CC=gcc MYKMEANS_NATIVE=1 python -m pip install .
```

`MYKMEANS_NATIVE=1`は対応するコンパイラで`-march=native`を有効にする。通常のビルドでは無効。異なるCPUへ配布するバイナリには使用せず、手元のマシンで通常ビルドと比較する。

Linux / WindowsはOpenMP有効、macOSはシリアルビルドを標準にする。macOSでOpenMPを使う場合は`libomp`とコンパイラのインクルード／リンクパスを準備し、`MYKMEANS_OPENMP=1`を指定する。OpenMPなしでもCythonのネイティブコードは使える。

```bash
MYKMEANS_OPENMP=0 python -m pip install .
```

Windowsの環境変数はPowerShell等の構文で設定する。CコンパイラはVisual Studio Build Toolsを使用する。OSごとのCI定義を同梱するが、定義の存在は各OSでの実行実績を意味しない。実際に検証した環境は[検証結果](docs/validation.md)に記録する。

## 最小例

```python
import numpy as np
from mykmeans import KMeans

X = np.random.default_rng(0).normal(size=(100_000, 8))

km = KMeans(
    n_clusters=16,
    algorithm="hamerly",
    n_init="auto",
    random_state=0,
    n_threads=4,
).fit(X)

print(km.inertia_, km.n_iter_, km.backend_)
labels = km.predict(X)
distances = km.transform(X[:10])  # ユークリッド距離、二乗ではない
```

既存コードの基本的な置換はimportの変更で行える。

```python
# from sklearn.cluster import KMeans
from mykmeans import KMeans
```

標準引数と既定値はscikit-learnの公開APIに合わせる。追加引数は`n_threads`と`record_history`。`algorithm`には標準の`"lloyd"`、`"elkan"`に加え、`"hamerly"`、`"exponion"`、`"auto"`を指定できる。計算結果のビット単位一致や、すべての周辺機能との完全互換を意味するものではない。対応範囲は[互換性](docs/compatibility.md)を参照。

## 実行方式

| 入力・指定 | 実行方式 |
| --- | --- |
| 密行列、`algorithm="lloyd"` | 独自Cython。小さい次元・クラスタ数では直接距離計算、それ以外ではブロックBLASと候補の直接評価 |
| 密行列、`algorithm="hamerly"` | 独自Cython。点ごとの上下界を使う三角不等式の枝刈り |
| 密行列、`algorithm="elkan"` | 独自Cython。点と各中心の下界、および中心間距離を使う枝刈り |
| 密行列、`algorithm="exponion"` | 独自Cython。Hamerlyの上下界とExponionの候補球。中心の近傍順序には全ソートを使う変種 |
| 密行列、`algorithm="auto"` | 64次元以下かつクラスタ数が2以上ならHamerly、それ以外はLloyd |
| CSR疎行列、`algorithm="lloyd"` | 独自CSR Cython。入力全体を密行列へ展開せずに割当・重心更新 |
| CSR疎行列、`algorithm="elkan"` | 独自CSR Cython。CSR距離計算とElkanの上下界 |
| CSR疎行列、`algorithm="auto"` | CSR Lloyd |

他形式のSciPy疎行列はCSRへ変換する。CSR入力で`"hamerly"`または`"exponion"`を指定すると`ValueError`になる。`auto`は固定の選択規則で、速度の最適性は保証しない。`backend_`、`algorithm_`、`assignment_engine_`で実際の経路を確認できる。

Elkanはfloat64の`n_samples × n_clusters`下界配列を持つ。Exponionは`n_clusters × n_clusters`の距離・近傍順序を持ち、各反復のソートに`O(k² log k)`の時間がかかる。距離計算回数を減らせても、メモリ転送やソートの負担で遅くなる条件がある。

## 重みと初期化

```python
weights = np.linspace(0.1, 2.0, len(X))
km.fit(X, sample_weight=weights)

# 初期中心を固定して実装・アルゴリズムを比較する場合
km = KMeans(16, init=X[:16].copy(), n_init=1, tol=0).fit(X)
```

- `init`: 独自のweighted greedy k-means++、重み付きランダム抽出、中心配列、callable。
- `n_init`: 各開始の最終目的関数が最小の結果を採用。`"auto"`はk-means++と中心配列で1回、ランダムとcallableで10回。
- 密行列のk-means++は、三角不等式から改善不能と判断できる距離計算を省略する。小規模データや剪定率の低い候補では通常の直接計算を選ぶ。この選択はsampling分布とgreedy local trialsを変更しない。
- ゼロ重みは許す。負・非有限・合計ゼロの重みは拒否する。
- 同じseedでもscikit-learnとのビット単位一致は保証しない。
- 密行列の`copy_x=False`は、利用可能な入力配列を一時的に中心化して戻す。丸めによる微小な変化はあり得る。CSRは中心化せず、中心は密行列として保持する。
- 各開始の最後に、実際に保存する中心からラベルと目的関数を再評価する。密行列では元座標への復元による丸めも含め、最終目的関数で開始結果を比較する。

## 数学向け機能

高価な解析は明示的に呼び出す。通常の`fit()`で共分散や評価指標を計算しない。

| 関数・メソッド | 内容 |
| --- | --- |
| `cluster_statistics` / `km.diagnostics` | 重み付き平均、半径、クラスタ別SSE、共分散、ANOVA分解 |
| `km.transform_squared(X)` | 全中心への二乗距離。密行列・CSRともfloat64出力 |
| `km.predict_distances(X)` | 最近傍中心への二乗距離。float64出力 |
| `objective_gradient(X, centers, labels=...)` | 所属を固定した目的関数の勾配 |
| `assignment_hessian(labels, k, d, sample_weight=...)` | ブロック対角Hessian。標準ではCSR |
| `voronoi_halfspaces(centers, index)` | Voronoiセルを半空間の連立不等式で取得 |
| `voronoi_halfspaces(..., power_weights=...)` | 加法的重みを持つpower diagram |
| `FeatureMetric(feature_weights=...)` | 特徴量ごとの重みをEuclidean座標へ変換 |
| `FeatureMetric(metric=M)` | 正半定値の二次形式／Mahalanobis型計量の変換 |
| `align_centers` / `align_labels` | 最適な対応づけによるラベル置換の除去 |
| `soft_assign` / `km.soft_assign` | 温度付きGibbs重み。校正済み確率とは解釈しない |
| `evaluate_k` | kごとのinertia、silhouette、Calinski–Harabasz、Davies–Bouldin |
| `clustering_stability` | 初期値を変えた分割のARI比較 |
| `gap_statistic` | 一様参照分布を用いたgap統計量と選択規則 |
| `adjusted_rand_score` | ラベルの置換に不変なadjusted Rand index |
| `silhouette_samples` / `silhouette_score` | 各点／平均のEuclidean silhouette |
| `calinski_harabasz_score` | クラスタ間分散とクラスタ内分散の比 |
| `davies_bouldin_score` | クラスタ内散布と中心間距離による指標 |

```python
from mykmeans import (
    objective_gradient, assignment_hessian, FeatureMetric, silhouette_score,
)

stats = km.diagnostics(X, covariance="diag")
gradient = objective_gradient(X, km.cluster_centers_, labels=km.labels_)
hessian = assignment_hessian(km.labels_, km.n_clusters, X.shape[1])

metric = FeatureMetric(feature_weights=[1.0] * X.shape[1])
transformed_X = metric.fit_transform(X)
quality = silhouette_score(X, km.labels_, sample_size=2000, random_state=0)
```

評価指標5関数も独立実装。ARI、silhouette、Calinski–Harabasz、Davies–Bouldinは非重み付きで、学習に与えた重みを自動適用しない。Silhouetteは距離行列を分割して処理するが、全点評価の演算量は二乗のまま。必要に応じて`sample_size`を指定する。数学用の解析関数と座標を必要とする評価指標は密行列を対象にする。`km.soft_assign`はCSR入力でも利用できるが、出力は密な`n × k`行列になる。

数式、戻り値、計算量、未収束時の解釈は[数学機能](docs/mathematics.md)、全体の仕様は[API](docs/api.md)を参照。

## scikit-learnとの連携

パラメータ管理、入力検証、clone用フック、出力形式の処理は独自の`_compat`基盤に実装した。単独の学習・予測・解析にscikit-learnは不要。PipelineやGridSearchCV、metadata routingと連携するときは追加でインストールする。

```bash
python -m pip install ".[sklearn]"
```

```python
from sklearn.pipeline import make_pipeline
from mykmeans import FeatureMetric, KMeans

pipeline = make_pipeline(
    FeatureMetric(feature_weights=[1.0] * X.shape[1]),
    KMeans(16, algorithm="exponion", random_state=0),
)
pipeline.fit(X)
```

連携時だけscikit-learnのタグやmetadata requestのオブジェクトを利用する。`set_output(transform="pandas")`にはpandas、`"polars"`にはpolarsを別途用意する。これらも数値計算の必須依存には含まれない。

## 履歴・実行情報

```python
km = KMeans(16, record_history=True, random_state=0).fit(X)
print(km.restart_inertias_)
print(km.convergence_reason_)
print(km.distance_evaluations_)
print(km.history_[-1])
```

密行列・CSRとも反復履歴と距離評価回数を取得できる。`history_`の`objective_fixed_assignment`は、その反復の所属を固定した更新後SSE。密行列では中心化座標上で評価する。最終再割当や元座標に戻す丸めの影響により、最終`inertia_`とは異なり得る。

`distance_evaluations_`は割当処理で評価した点と中心の組数。初期化、中心間距離、sorting、任意の履歴集計を含む総演算回数ではない。履歴とカウンタは選ばれた開始に対応する。

## 性能とメモリ

`float32`では入力・中心のメモリを減らし、距離の最終評価と重心集計はfloat64で行う。十分に分離したクラスタではHamerlyやExponionの枝刈りが有効になることがある。高次元や重なりの強いデータでは、ブロックBLASを使うLloydと比較する。

以下は枝刈りのための追加記憶量。入力、中心、ラベル、各workerの重心集計バッファなどは別に必要になる。

| 方式 | 枝刈りに使う主な追加記憶 |
| --- | --- |
| Lloyd | 枝刈り用の全点・全中心配列は持たない。密行列のBLAS一時領域はブロック単位 |
| Hamerly | 点ごとの上下界など`O(n)` |
| Elkan、CSR Elkan | float64の下界`8nk` bytesと、中心間距離など`O(k² + n)` |
| Exponion | 点ごとの上下界`O(n)`と、中心間距離・近傍順序`O(k²)` |

CSRは元の`n × d`データ全体を密行列に変換しないが、中心とworker別の集計には`O(kd)`、`O(Tkd)`の密な記憶領域を使う。非常に大きな特徴量数・クラスタ数では、この領域も確認する。

`transform`、`transform_squared`、`soft_assign`は出力そのものが`n × k`。`record_history=True`は追加の目的関数計算を行う。速度を比較するときは同じdtype、初期中心、停止条件、スレッド数で測り、目的関数と反復数も記録する。

BLASの内部並列は抑制し、OpenMP側で並列化する。外側で`GridSearchCV(n_jobs=...)`等を並列化する場合は、内側の`n_threads=1`も比較する。結果表と再実行手順は[ベンチマーク](benchmarks/README.md)にまとめる。

## 検証と公開

```bash
python -m pip install ".[dev]"
python -m pytest -q
python examples/basic.py
python examples/research.py
python benchmarks/benchmark.py --quick --threads 1 --repeats 3 --dtypes float32,float64 --output benchmarks/results.json
python -m build
```

実行環境と検証範囲は[検証結果](docs/validation.md)を参照。ソース、テスト、ベンチマーク、CI定義、MITライセンスを含む。生成された`.so`や特定マシン用バイナリをGitに追加しない。`.gitignore`を同梱している。

## 参考文献

- [scikit-learn KMeans API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Elkan, Using the Triangle Inequality to Accelerate k-Means](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf)
- [Hamerly, Making k-means even faster](https://cs.baylor.edu/~hamerly/papers/sdm_2010.pdf)
- [Newling & Fleuret, Fast k-Means with Accurate Bounds](https://proceedings.mlr.press/v48/newling16.html)
- [Arthur & Vassilvitskii, k-means++](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
- [Cython: Using Parallelism](https://cython.readthedocs.io/en/stable/src/userguide/parallelism.html)

2025–2026年の高速化研究、GPU方式、厳密計算と近似計算の区別は[高速化手法の調査](docs/sota.md)を参照。

## ライセンス

[MIT](LICENSE)。アルゴリズムの数学的内容は原論文を参照し、実装は独立に記述した。依存ライブラリはそれぞれのライセンスに従う。
