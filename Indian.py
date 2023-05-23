from tensorflow.keras import optimizers

class Indian(optimizers.Optimizer):
    def __init__(self, learning_rate = 0.01, alpha = 0.5, beta = 0.1, name = "Indian"):
        super().__init__(name = name)
        # 学習率の設定には _build_learning_rate 関数を用いる
        self._learning_rate = self._build_learning_rate(learning_rate)
        self._alpha = alpha
        self._beta = beta
    
    # build 関数は保持する変数を定義する関数
    def build(self, var_list):
        
        super().build(var_list)
        # build 関数はイテレーション毎に呼び出されるので, 初めの1度のみ処理されるようにする
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

        # 
        self._y = list()
        for variable in var_list:
            self._y.append(
                self.add_variable_from_reference(
                    model_variable = variable, variable_name = "y",
                    # 初期値は0
                    initial_value = tf.zeros(shape = variable.shape)
                )
            )
    
    def update_step(self, gradient, variable):
        # ハイパーパラメータはパラメータと同じデータ型にキャストする
        learning_rate = tf.cast(self._learning_rate, variable.dtype)
        alpha = tf.cast(self._alpha, variable.dtype)
        beta = tf.cast(self._beta, variable.dtype)

        # 保持する変数を取得するためのキー
        var_key = self._var_key(variable)

        # yを取得
        y = self._y[self._index_dict[var_key]]
        
        # 更新前のパラメータを保持
        tmp_variable = variable.value()
        # パラメータは assign 関数で更新する
        variable.assign( variable + learning_rate * ( ((1/beta) - alpha) * variable - (1/beta) * y - beta * gradient ) )
        # yを更新
        y.assign( y + learning_rate * ( ((1/beta) - alpha) * tmp_variable - (1/beta) * y ) )
