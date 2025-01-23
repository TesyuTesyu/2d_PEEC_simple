・遅延を考慮していない2dの (R, L, C) PEEC。
・自由空間における導体版のシミュレーション。
・形状は画像から生成される。黒色が導体の領域、白色がなにもない領域となる。
・python 3.12.1 VS-codeで動作.
![スクリーンショット 2025-01-23 202045](https://github.com/user-attachments/assets/3893ac29-1125-4f18-98ea-8709d5bf5cd8)

参考：
[1] Ekman, Jonas, "Electromagnetic modeling using the partial element equivalent circuit method," Luleå tekniska universitet,  Doctoral thesis, 2003.

X: @Testes_int

[動かないときに考えられること]：
・numpy, matplotlibなどのライブラリはpipなどでインストールしてください。
・電圧源を特定の座標に設定することができますが、それが導体上にないときはエラーになります（ただし判定していないので変なエラー）。電圧源は一つ目に表示れる画像で白く表示され、それが指定した個数あるかどうかを確認できます。
・画像を読み込む際、セキュリティソフトが邪魔をすることがあります。

[問題点]：
・mutual L,Cは全面積で計算されている.
　相互インダクタンスやキャパシタンスは十分に遠い距離であれば無視できるはずです。カットオフ距離を設定し、ある範囲でのみ計算すると軽くなるかもしれません。未実装。
・本来、物体の輪郭の微小面は面積を半分にするが、このプログラムは全て同じ面積で処理されている。
　L, Cを計算するとき、厳密には上記のように条件分岐して計算すべきです。いつか実装したいです。
・電流のグラフが見にくい. 例えば電流をベクトル表示したい.

まだおもちゃの段階ですが、だからこそ比較的簡単に弄ることができるはずです。参考になれば幸いです。
また、バグがあれば教えてほしいです。
