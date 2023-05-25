from tensorflow.keras import optimizers

from tensorflow.keras import optimizers

class AdaBelief(optimizers.Optimizer):
    def __init__(self, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7,name = "AdaBelief"):
        super().__init__(name = name)
        # 学習率の設定には _build_learning_rate 関数を用いる
        self._learning_rate = self._build_learning_rate(learning_rate)
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
    
    # build 関数は保持する変数を定義する関数
    def build(self, var_list):
        
        super().build(var_list)
        # build 関数はイテレーション毎に呼び出されるので, 初めの1度のみ処理されるようにする
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

        # 一次モーメントを保持する変数
        self._m = list()
        # 二次モーメントを保持する変数
        self._s = list()
        for variable in var_list:
            self._m.append(
                self.add_variable_from_reference(
                    model_variable = variable, variable_name = "m",
                    # 初期値はパラメータと同じにする
                    initial_value = tf.zeros(shape = variable.shape)
                )
            )
            self._s.append(
                self.add_variable_from_reference(
                    model_variable = variable, variable_name = "s",
                    # 初期値はパラメータと同じにする
                    initial_value = tf.zeros(shape = variable.shape)
                )
            )
    
    def update_step(self, gradient, variable):
        # ハイパーパラメータはパラメータと同じデータ型にキャストする
        learning_rate = tf.cast(self._learning_rate, variable.dtype)
        beta_1 = tf.cast(self._beta_1, variable.dtype)
        beta_2 = tf.cast(self._beta_2, variable.dtype)
        epsilon = tf.cast(self._epsilon, variable.dtype)
        # イテレーション数を取得
        iteration = tf.cast(self.iterations + 1, variable.dtype)

        # 保持する変数を取得するためのキー
        var_key = self._var_key(variable)

        # 一次モーメントを取得
        m = self._m[self._index_dict[var_key]]
        # 二次モーメントを取得
        s = self._s[self._index_dict[var_key]]
        
        # 一次モーメントを更新
        m.assign(beta_1 * m + (1 - beta_1) * gradient)
        # 二次モーメントを更新
        s.assign(beta_2 * s + (1 - beta_2) * (gradient - m) * (gradient - m))

        # 補正項の計算
        m_hat = m / (1 - tf.math.pow(beta_1, iteration))
        s_hat = s / (1 - tf.math.pow(beta_2, iteration))

        # パラメータは assign 関数で更新する
        variable.assign( variable - learning_rate * m_hat / (tf.math.sqrt(s_hat) + epsilon) )
