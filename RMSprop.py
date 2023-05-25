import tensorflow as tf
from tensorflow.keras import optimizers

class RMSprop(optimizers.Optimizer):
    def __init__(self, learning_rate = 0.001, rho = 0.9, epsilon = 1e-7,name = "RMSprop"):
        super().__init__(name = name)
        # 学習率の設定には _build_learning_rate 関数を用いる
        self._learning_rate = self._build_learning_rate(learning_rate)
        self._rho = rho
        self._epsilon = epsilon
    
    # build 関数は保持する変数を定義する関数
    def build(self, var_list):
        
        super().build(var_list)
        # build 関数はイテレーション毎に呼び出されるので, 初めの1度のみ処理されるようにする
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

        # 二次モーメントを保持する変数
        self._v = list()
        for variable in var_list:
            self._v.append(
                self.add_variable_from_reference(
                    model_variable = variable, variable_name = "v",
                    # 初期値はパラメータと同じにする
                    initial_value = tf.zeros(shape = variable.shape)
                )
            )
    
    def update_step(self, gradient, variable):
        # ハイパーパラメータはパラメータと同じデータ型にキャストする
        learning_rate = tf.cast(self._learning_rate, variable.dtype)
        rho = tf.cast(self._rho, variable.dtype)
        epsilon = tf.cast(self._epsilon, variable.dtype)

        # 保持する変数を取得するためのキー
        var_key = self._var_key(variable)

        # 二次モーメントを取得
        v = self._v[self._index_dict[var_key]]
        
        # 二次モーメントを更新
        v.assign(rho * v + (1 - rho) * gradient * gradient)
        # パラメータは assign 関数で更新する
        variable.assign( variable - learning_rate * gradient / (tf.math.sqrt(v) + epsilon) )
