from tensorflow.keras import optimizers

class AdaGrad(optimizers.Optimizer):
    def __init__(self, learning_rate = 0.001, epsilon = 1e-7,name = "AdaGrad"):
        super().__init__(name = name)
        # 学習率の設定には _build_learning_rate 関数を用いる
        self._learning_rate = self._build_learning_rate(learning_rate)
        self._epsilon = epsilon
    
    # build 関数は保持する変数を定義する関数
    def build(self, var_list):
        
        super().build(var_list)
        # build 関数はイテレーション毎に呼び出されるので, 初めの1度のみ処理されるようにする
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

        # 勾配の二乗を累積する変数
        self._h = list()
        for variable in var_list:
            self._h.append(
                self.add_variable_from_reference(
                    model_variable = variable, variable_name = "h",
                    # 初期値はパラメータと同じにする
                    initial_value = tf.zeros(shape = variable.shape)
                )
            )
    
    def update_step(self, gradient, variable):
        # ハイパーパラメータはパラメータと同じデータ型にキャストする
        learning_rate = tf.cast(self._learning_rate, variable.dtype)
        epsilon = tf.cast(self._epsilon, variable.dtype)

        # 保持する変数を取得するためのキー
        var_key = self._var_key(variable)

        # 勾配の二乗の累積を取得
        h = self._h[self._index_dict[var_key]]
        
        # 勾配の二乗の累積を更新
        h.assign(h + gradient * gradient)
        # パラメータは assign 関数で更新する
        variable.assign( variable - learning_rate * gradient / (tf.math.sqrt(h) + epsilon) )
