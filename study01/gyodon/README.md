# 機械学習 牛丼判別(Linux or Mac or Bash on Windows)

## もくじ
* 機械学習 準備
* 機械学習 学習
* 画像判別
* 機械学習 精度向上
* 忙しい人のための画像データ

## 機械学習 準備

### コードをダウンロード
書籍「[Pythonによるスクレイピング＆機械学習 開発テクニック BeautifulSoup、scikit-learn、TensorFlowを使ってみよう](http://www.socym.co.jp/support/s-1079)」

上記ページの「初版1刷～5刷用サンプルプログラム」からサンプルプログラムをダウンロードする。その後、解凍したファイル内の「ch7」を使用する。

### Python3のインストール

Python3のインストール方法は各種存在しますが、ここではAnacondaでの環境構築を紹介します。

Anaconda は、Continuum Analytics 社によって提供されている、Python 本体に加え、科学技術、数学、エンジニアリング、データ分析など、よく利用される Python パッケージ（2016 年 2 月時点で 400 以上）を一括でインストール可能にしたパッケージです。

https://github.com/redimpulz/ml_hack/issues/1

### pipでライブラリをインストール

pipは、pythonのパッケージ管理システムです。

Python 2.7.9以降、Python 3.4以降のバージョンにはデフォルトでインストールされています。
インストールされていない場合は、以下を参考にインストールしてください。

[Pythonのパッケージ管理システムpipのインストールと使い方](http://uxmilk.jp/12691)

pipコマンドで必要なパッケージをインストールする。

```
$ pip install beautifulsoup4
$ pip install sklearn
$ pip install h5py
$ pip install keras
```

### プロキシの設定

UECWirelessを使用している場合、本プログラムの実行にはプロキシ設定が必要になる。
以下のコマンドをターミナル等に入力してから、ダウンロードを実行する。

#### Mac, Linuxの場合

ターミナルから、以下のコマンドを入力

```
$ export HTTP_PROXY=http://proxy.uec.ac.jp:8080
$ export HTTPS_PROXY=http://proxy.uec.ac.jp:8080
```

#### Windowsの場合

コマンドプロンプトから、以下のコマンドを入力

```
$ set HTTP_PROXY=http://proxy.uec.ac.jp:8080
$ set HTTPS_PROXY=http://proxy.uec.ac.jp:8080
```

### 画像取得
はじめに学習データとなる牛丼画像をスクレイピングで取得する。
```
$ python3 gyudon_downloader.py
```

72行目
```
download_all("牛丼", "./image/gyudon")
```
にて検索ワードと取得枚数の指定が可能。デフォルトは「牛丼」で検索した画像1000枚を取得する。

例：チーズ牛丼の画像を100枚
```
download_all("チーズ牛丼", "./image/gyudon", 100)
```

### 画像の振り分け
つぎにダウンロードした牛丼画像の振り分けを行う。スクレイピングで取得した画像はimageディレクトリに保存されていて、これらの画像を手作業で各ディレクトリに移動させ画像のラベルづけを行う。
```
 ch7/
    ├ gyudon_downloader.py
    ├ image
        ├ gyudon --- 未分類の牛丼画像
        ├ normal --- 通常の牛丼
        ├ beni   --- 紅しょうが牛丼
        ├ negi   --- ねぎ玉牛丼
        ├ cheese --- チーズ牛丼
        ├ kimuti --- キムチ牛丼
        └ other  --- その他の画像、牛丼じゃない画像
```

ドラッグ&ドロップで振り分けるなど方法は様々だが、PHPを使用して振り分ける方法を紹介する。まずPHPでローカルサーバーを起動する。
```
$ php -S localhost:8000 -t ch7/
```

実行後ch7がルートディレクトリになり、ブラウザから[localhost:8000/gyudon-hand.php](localhost:8000/gyudon-hand.php)へアクセスできるようになる。ブラウザに表示された画像が何牛丼なのか振り分ける。移動はPHPが行ってくれる。

## 機械学習 学習

### 画像を数値データに変換
python の Numpy を使用して振り分けた画像を元に数値データを作成する。

```
$ python3 gyudon-makedata.py
--- normal を処理中
--- beni を処理中
--- negi を処理中
--- cheese を処理中
ok, 300
```

gyudon-makedata.py を実行すると「image/gyudon.npy」という Numpy のデータが作成される。

### CNNで学習
Numpyのデータを畳み込みニュートラルネットワーク(CNN)で学習させる。
```
$ python3 gyudon_keras.py
Using TensorFlow backend.
...
loss= 0.86482
accuracy= 0.90133
```
この場合、正解率は0.865(85%)を意味する。

学習に5分程かかる。

## 画像判別
学習させたモデルを元に、画像を判別する。

```
$ python3 gyudon-checker.py (画像パス1) (画像パス2) ...
```
上記コマンドの実行、解析結果HTMLが生成される。

```
 ch7/
    ├ gyudon-result.html
```

## 機械学習 精度向上
正解率を上げるために、チューニングを行う。画像の角度を変えたり反転させたりしてデータ数を増やす。

先ほど使用した gyudon-makedata.py を改良したものが gyudon-makedata2.py になる。コマンドから以下を実行して画像データを水増しする。
```
$ python3 gyudon-makedata2.py
ok, 2020
```

画像データが増えた状態で学習させる。
```
$ python3 gyudon_keras2.py
```
学習に20分程かかる。

画像判別は、前回同様のコマンドを実行する。
```
$ python3 gyudon-checker.py (画像パス1) (画像パス2) ...
```

## 忙しい人のための画像データ
画像を振り分けする時間がないときのための振り分け済画像は以下のリンクからダウンロードできる。

[画像データ](https://drive.google.com/file/d/0B4TRTBPCoa6zdHdhcUhEU2hrcFE/view?usp=sharing)