import tensorflow as tf
from tensorflow.keras import optimizers

class SGD(optimizers.Optimizer):
    def __init__(self, learning_rate = 0.01, name = "SGD"):
        super().__init__(name = name)
        # 学習率の設定には _build_learning_rate 関数を用いる
        self._learning_rate = self._build_learning_rate(learning_rate)
    
    # build 関数は保持する変数を定義する関数
    def build(self, var_list):
        
        super().build(var_list)
        # build 関数はイテレーション毎に呼び出されるので, 初めの1度のみ処理されるようにする
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
    
    def update_step(self, gradient, variable):
        # ハイパーパラメータはパラメータと同じデータ型にキャストする
        learning_rate = tf.cast(self._learning_rate, variable.dtype)

        # パラメータは assign 関数で更新する
        variable.assign( variable - learning_rate * gradient )
