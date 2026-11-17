# 検証記録 — mykmeans 0.2.0

実行日: **2026-11-17**。実装、テスト、パッケージ、性能を分けて記録する。CI 定義の存在を、他の OS で実行済みであることとは扱わない。

## 実行環境

Linux x86_64、Python 3.12.14、GCC 14.2、Cython 3.3.1。主なテスト・性能計測環境は NumPy 2.3.5、SciPy 1.17.0、threadpoolctl 3.6.0。scikit-learn は比較・任意のプロトコル連携テストのみに導入した。

`_core.pyx` と `_sparse.pyx` の両拡張をコンパイルした。通常ビルドは portable な `-O3 -std=c99 -fopenmp` で、`-march=native` と fast-math は使わない。

## テストスイート

| 実行条件 | 結果 |
| --- | --- |
| OpenMP 有効、scikit-learn 1.6.1 | **259 passed** |
| OpenMP 有効、scikit-learn 1.9.1 | **259 passed** |
| 別のソースコピーから OpenMP 無効で両拡張を再ビルド、scikit-learn 1.9.1 | **259 passed** |

同一スイートを複数条件で動かした値であり、777 種類の異なるテストがあるという意味ではない。

```bash
python -m pip install ".[test]"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

主な検証対象:

- 密行列の Lloyd / Hamerly / Elkan / Exponion、CSR の Lloyd / Elkan。
- float32 / float64、重み付き・ゼロ重み、複数初期化、手動初期中心、callable 初期化。
- NumPy の直接差分での最近傍と目的関数、同じ初期中心からの独立 Lloyd との比較。
- 重複点、空クラスタ、同距離、単一クラスタ、早期停止、大きい共通オフセット。
- 公開する最終中心と元座標によるラベル・inertia の再評価。
- 1 / 4 スレッド、読み取り専用・非連続入力、copy_x と例外時の復元。
- 正規化されていない CSR、不正な CSR 添字の拒否、観測行列を密行列へ変換しないこと。
- 初期化フィルタの直接計算との一致、境界の隣接浮動小数点数、極小・極大スケール。
- ANOVA 分解、共分散、固定割当の勾配・Hessian、PSD 計量、Voronoi / power diagram。
- 独立に実装した ARI、silhouette、Calinski–Harabasz、Davies–Bouldin と定義式・参照値の比較。
- clone、Pipeline、GridSearchCV、DataFrame の名前・順序・index、出力設定、sample_weight の metadata routing。

## scikit-learn からの独立性

`tests/test_independent.py` は新しい Python プロセスで sklearn 名前空間の import を全面的に禁止する。その状態で全学習方式、密行列・CSR の初期化、予測、距離、score、数学機能、独立 metrics、k の比較、初期値安定性、gap statistic を実行する。数値処理中の sklearn import の試行自体がゼロであることを検査した。

Python ソースの import も監査し、任意連携用の `Tags` 等と `MetadataRequest` 以外の sklearn import がないことを確認した。これらは API の情報を受け渡す値オブジェクトで、数値計算を行わない。推定器の基盤と入力検証もこのパッケージ内の独立実装。

さらに、**scikit-learn がインストールされていない新規 venv** に、ビルドした wheel と必要な依存だけをインストールして確認した。

| 項目 | 結果 |
| --- | --- |
| インストールした wheel | `mykmeans-0.2.0-cp312-cp312-linux_x86_64.whl` |
| venv の依存 | NumPy **2.5.5** / SciPy **1.18.1** / threadpoolctl **3.8.0** |
| sklearn の検出 | `importlib.util.find_spec("sklearn") is None` |
| インストール済みパッケージの数値 API 確認 | 全方式・両 dtype・CSR・数学 API・モデル選択を実行成功 |
| `examples/basic.py` / `examples/research.py` | 実行成功 |

ソースツリーを `PYTHONPATH` に入れず、venv の `site-packages` から読み込んだ。任意の sklearn 連携例 `examples/sklearn_integration.py` も、sklearn 1.9.1 環境で実行した。

## sklearn の汎用 estimator checks

独自テスト259件とは別に、scikit-learn 1.9.1 が生成する estimator checks を実行した。

| 推定器 | 合計 | 成功 | 不一致 | skip |
| --- | ---: | ---: | ---: | ---: |
| `FeatureMetric` | 47 | 46 | 0 | 1 |
| `KMeans` | 55 | 52 | 2 | 1 |

両方の skip は、`SCIPY_ARRAY_API` 未設定による `check_array_api_input`。Array API デバイス配列は本リリースの対応範囲に含めない。

KMeans の2件は、整数重みを使う学習と標本を明示的に複製する学習との、dense / sparse の `check_sample_weight_equivalence`。独立初期化でクラスタ番号の順序が置換され、番号そのものの比較で不一致になった。

- 両方の inertia は `2.9098625181736644` で同じ。
- 予測分割の ARI は **1.0**。
- 中心の順序を対応づけるとラベルは完全一致。
- 整列した中心の最大差は dense で `1.1102230246251565e-16`、CSR で `0.0`。

したがって「sklearn の全チェックに通過」とは記載しない。インターフェイスの互換性と、異なる初期化手順でのクラスタ番号の完全一致を区別する。[互換性の詳細](compatibility.md)を参照。

## 性能計測と目的関数の監査

密行列96条件×2スレッド設定とCSR16条件、計208行の実測を保存した。各行に warmup 1 回を除く3回の計測値を持つ。全条件で入力生成と独立の目的関数検算を計測外に置いた。

- float64 の固定初期中心条件では全方式で ARI=1.0、反復回数一致。
- float32 の固定初期中心 `uniform_2d` は sklearn との ARI 約0.998189、独立 SSE 差約+0.003073%。この差を削除せず記録した。
- mykmeans の報告 inertia と、別途計算した直接差分 SSE の最大相対誤差は約 **3.321e-14**。
- CSR の今回の条件は sklearn Lloyd より遅い。速い条件だけを抜き出して優位を主張しない。
- 各 JSON のソース SHA-256 が、配布する `src/mykmeans` の対応ファイルと一致することを確認した。

表・全結果・再現コマンドは [ベンチマーク記録](../benchmarks/README.md) にある。研究上の SoTA に関する一次資料の調査と採用判断は [事前調査](sota.md) に分けた。

## 配布範囲と未検証の範囲

GitHub 向け ZIP は `mykmeans/` をルートとするソース一式で、ライセンス、ビルド設定、テスト、CI、文書、例、計測コードと JSON を含む。ローカルの `.so`、ビルドキャッシュ、venv、CPU 専用バイナリは含めない。pip 用 wheel のビルドとインストールは上記で確認した。

作成した ZIP を新しいディレクトリへ展開し、そこから両 Cython 拡張を含む wheel を再ビルドできることも確認した。ZIP の CRC、相対パス、CI 定義・ヘッダ・両 `.pyx` の収録、独立配布する Qiita 記事との内容一致を検査した。

Windows / macOS、Python の全バージョン、すべての NumPy / SciPy / sklearn バージョン組合せ、すべての CPU / BLAS について実行したわけではない。OS 別の CI 定義を同梱したが、この記録の実測は Linux。GPU・分散・ストリーミング・大域的最適解の保証は対象外。
