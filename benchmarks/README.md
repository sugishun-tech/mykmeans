# 再現可能な性能比較 — mykmeans 0.2.0

測定日: 2026-11-17。以下はこのソースを実際にビルドして測った値。速度比は **scikit-learn Lloyd の時間 ÷ 比較対象の時間** とし、1 未満は比較対象のほうが遅い。

## 環境と手順

- Linux x86_64 / AMD EPYC 9V74、利用可能な論理 CPU は 9。
- Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0、scikit-learn 1.9.1、Cython 3.3.1、GCC 14.2。
- portable build: `-O3 -std=c99 -fopenmp`。`-march=native` と fast-math は使わない。
- dense は 1 / 4 スレッド。CSR は 1 スレッド。`threadpoolctl` と `n_threads` の設定を揃えた。mykmeans 内部では BLAS を 1 スレッドに制限し、OpenMP の外側並列を使う。
- `n_init=1, max_iter=100, tol=1e-4`。各方式・各条件を 1 回ウォームアップし、その後 3 回の **fit 全体** を計測。表は中央値、単位は ms。
- データ生成、warmup、独立 SSE 計算、ARI 計算を計測に含めない。各反復の入力検証・中心化・最終再割当は fit 内なので含まれる。
- `fixed` は同一の明示的な初期中心。`k-means++` は各実装が初期化から実行し、同じ random_state でも中心列・反復回数が一致するとは限らない。
- 同時に別のビルド・重いテスト・他のベンチマークを実行しなかった。専用の物理マシンを占有した測定ではなく、短い時間の値には環境の揺れがある。3 回の測定から強い統計的結論は出さない。

## データ

| 名前 | n | d | k | 生成条件 |
| --- | ---: | ---: | ---: | --- |
| separated_3d | 25,000 | 3 | 32 | make_blobs、std=0.65、center_box=(-20,20) |
| overlapping_16d | 12,000 | 16 | 16 | make_blobs、std=7.0、同じ center_box |
| highdim_128d | 5,000 | 128 | 32 | make_blobs、std=3.0、同じ center_box |
| uniform_2d | 30,000 | 2 | 64 | [-1,1]² の一様乱数 |
| csr_1000d_1pct | 12,000 | 1,000 | 16 | 非ゼロ密度 1%、格納値は標準正規分布 |
| csr_64d_10pct | 20,000 | 64 | 16 | 非ゼロ密度 10%、格納値は標準正規分布 |

名称は生成条件を識別するためのもの。幾何学的な分離度を厳密に測るラベルではない。

## 密行列: 1 スレッド

全 16 条件を掲載する。SK は scikit-learn、MY は mykmeans。

| データ | dtype | 初期化 | SK Lloyd | SK Elkan | MY Lloyd | MY Hamerly | MY Elkan | MY Exponion |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| separated_3d | float32 | fixed | 9.047 | 17.309 | 12.607 | 11.630 | 21.036 | 9.971 |
| separated_3d | float32 | k-means++ | 32.329 | 34.000 | 18.025 | 18.611 | 21.203 | 18.728 |
| separated_3d | float64 | fixed | 10.723 | 16.925 | 9.398 | 9.923 | 20.741 | 9.148 |
| separated_3d | float64 | k-means++ | 18.454 | 28.686 | 20.063 | 19.064 | 22.109 | 19.517 |
| overlapping_16d | float32 | fixed | 7.100 | 19.694 | 13.528 | 16.265 | 21.804 | 15.347 |
| overlapping_16d | float32 | k-means++ | 21.432 | 40.394 | 39.845 | 22.533 | 36.775 | 20.746 |
| overlapping_16d | float64 | fixed | 8.965 | 19.944 | 13.908 | 14.518 | 22.155 | 15.542 |
| overlapping_16d | float64 | k-means++ | 21.828 | 31.927 | 34.809 | 19.315 | 40.014 | 18.156 |
| highdim_128d | float32 | fixed | 7.992 | 14.244 | 11.645 | 20.004 | 37.697 | 16.715 |
| highdim_128d | float32 | k-means++ | 33.888 | 36.412 | 39.764 | 36.798 | 44.478 | 36.900 |
| highdim_128d | float64 | fixed | 15.146 | 19.799 | 14.871 | 20.944 | 40.512 | 21.517 |
| highdim_128d | float64 | k-means++ | 22.480 | 28.029 | 40.542 | 36.990 | 44.521 | 36.802 |
| uniform_2d | float32 | fixed | 107.617 | 188.459 | 142.488 | 61.287 | 345.549 | 42.882 |
| uniform_2d | float32 | k-means++ | 155.348 | 216.933 | 161.445 | 74.205 | 295.204 | 62.903 |
| uniform_2d | float64 | fixed | 124.955 | 198.862 | 114.039 | 60.058 | 400.120 | 43.120 |
| uniform_2d | float64 | k-means++ | 144.610 | 221.256 | 126.316 | 69.096 | 338.727 | 58.459 |

## 密行列: 4 スレッド

全 16 条件を掲載する。SK は scikit-learn、MY は mykmeans。

| データ | dtype | 初期化 | SK Lloyd | SK Elkan | MY Lloyd | MY Hamerly | MY Elkan | MY Exponion |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| separated_3d | float32 | fixed | 7.167 | 12.156 | 6.980 | 5.738 | 8.838 | 6.238 |
| separated_3d | float32 | k-means++ | 32.882 | 37.399 | 14.380 | 14.904 | 16.576 | 14.555 |
| separated_3d | float64 | fixed | 8.104 | 24.259 | 7.676 | 6.621 | 12.996 | 5.916 |
| separated_3d | float64 | k-means++ | 19.894 | 22.496 | 13.980 | 14.455 | 16.691 | 15.991 |
| overlapping_16d | float32 | fixed | 6.850 | 16.171 | 7.871 | 7.273 | 10.071 | 9.078 |
| overlapping_16d | float32 | k-means++ | 18.893 | 37.079 | 16.647 | 15.085 | 21.226 | 11.973 |
| overlapping_16d | float64 | fixed | 7.081 | 15.886 | 7.508 | 8.408 | 16.616 | 7.400 |
| overlapping_16d | float64 | k-means++ | 14.663 | 43.881 | 16.680 | 11.235 | 17.922 | 9.325 |
| highdim_128d | float32 | fixed | 5.414 | 8.940 | 6.646 | 10.464 | 20.216 | 9.785 |
| highdim_128d | float32 | k-means++ | 69.682 | 77.279 | 18.781 | 21.761 | 22.572 | 21.456 |
| highdim_128d | float64 | fixed | 9.408 | 11.775 | 8.558 | 10.819 | 14.927 | 10.175 |
| highdim_128d | float64 | k-means++ | 16.109 | 17.277 | 18.606 | 17.697 | 21.817 | 17.451 |
| uniform_2d | float32 | fixed | 48.041 | 205.858 | 71.726 | 28.498 | 137.854 | 29.079 |
| uniform_2d | float32 | k-means++ | 107.074 | 225.349 | 87.368 | 50.201 | 145.746 | 44.409 |
| uniform_2d | float64 | fixed | 58.067 | 301.202 | 41.670 | 19.644 | 129.793 | 21.583 |
| uniform_2d | float64 | k-means++ | 89.549 | 233.133 | 57.456 | 38.505 | 118.488 | 41.187 |

## 疎行列: 1 スレッド、float64

分離した人工クラスタを追加せず、ランダムな疎構造で計測した。この条件では独立 CSR 実装は scikit-learn Lloyd より遅い。

| データ | 初期化 | SK Lloyd ms | SK Elkan ms | MY Lloyd ms | MY Elkan ms | MY Lloyd 倍率 | MY Elkan 倍率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| csr_1000d_1pct | fixed | 19.265 | 39.082 | 32.124 | 25.948 | 0.600× | 0.742× |
| csr_1000d_1pct | k-means++ | 38.734 | 65.925 | 49.913 | 47.461 | 0.776× | 0.816× |
| csr_64d_10pct | fixed | 76.379 | 119.664 | 145.163 | 120.763 | 0.526× | 0.632× |
| csr_64d_10pct | k-means++ | 63.127 | 99.634 | 110.219 | 90.435 | 0.573× | 0.698× |

## 品質と解釈

- float64 の固定初期中心条件では、dense / CSR とも全方式の ARI は 1.0。反復回数も一致した。浮動小数点の末尾まで同一の中心を保証する意味ではない。
- float32 の `uniform_2d` 固定中心条件では scikit-learn と mykmeans の ARI は **0.9981890634529433**、mykmeans の独立 SSE は相対で約 **+0.003073%**。同じ初期中心でも精度・集計順序・停止近傍の違いで分割がわずかに変わる。独立実装の4方式どうしは同じ分割になった。
- 初期化込みでは異なる局所解に至る場合がある。JSON の `inertia_independent`、`n_iter`、`ari_vs_sklearn_lloyd` を時間と合わせて確認する。
- 全208行のうち mykmeans の報告 inertia と別途計算した直接差分 SSE の最大相対誤差は **3.321e-14**。
- 1 スレッド・float64・uniform_2d・fixed では Exponion が **2.898×**。同じ条件の4スレッドでは Hamerly が **2.956×**、Exponion が **2.690×**。方式の順位はスレッド数でも変わった。
- 最も大きい倍率 **3.710×** は4スレッド・float32・highdim_128d・各自 k-means++ の Lloyd。これは初期化込みの比較で、共有初期中心による純粋な反復部分の倍率ではない。
- Elkan は距離計算を減らせても `n × k` 境界の更新が重く、今回の低次元条件では大きく遅い。距離評価回数だけを速度の代用にしない。
- `MYKMEANS_NATIVE=1` を使った0.2.0の比較結果は含めていない。最速を求める場合は、自分のデータとCPUでビルド条件・方式・スレッド数を変えて測る。

本測定は合成データのCPU比較。最新研究実装・GPU・分散システムとの直接比較は行っておらず、世界最速やSoTA達成を主張しない。事前調査と採用理由は [docs/sota.md](../docs/sota.md) に記載した。

## 再現コマンド

```bash
CC=gcc MYKMEANS_NATIVE=0 python -m pip install ".[dev]"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/benchmark.py --quick --threads 1 --repeats 3 --dtypes float32,float64 --output benchmarks/local_dense_1thread.json
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python benchmarks/benchmark.py --quick --threads 4 --repeats 3 --dtypes float32,float64 --output benchmarks/local_dense_4threads.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/benchmark_sparse.py --threads 1 --repeats 3 --output benchmarks/local_csr_1thread.json
```

`--quick` を外すと密行列の標本数は4倍になる。新しい計測は別ファイルに保存すると付属の記録を保持できる。

## 生データ

- [密行列・1 スレッド](results_portable_sklearn191_1thread.json): 96 行。
- [密行列・4 スレッド](results_portable_sklearn191_4threads.json): 96 行。
- [CSR・1 スレッド](results_csr_sklearn191_1thread.json): 16 行。

各行に3回の生時間、中央値・最小・最大、独立SSE、報告値の誤差、ARI、反復回数、実行経路、距離評価数を保存した。メタデータにバージョン、CPU、実効スレッド数、ビルド情報、計測時ソースのSHA-256を含む。
