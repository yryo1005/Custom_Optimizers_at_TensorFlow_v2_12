import tensorflow as tf
from tensorflow.keras import optimizers

class Momentum(optimizers.Optimizer):
    def __init__(self, learning_rate = 0.01, mu = 0.9, name = "Momentum"):
        super().__init__(name = name)
        # 学習率の設定には _build_learning_rate 関数を用いる
        self._learning_rate = self._build_learning_rate(learning_rate)
        self._mu = mu
    
    # build 関数は保持する変数を定義する関数
    def build(self, var_list):
        
        super().build(var_list)
        # build 関数はイテレーション毎に呼び出されるので, 初めの1度のみ処理されるようにする
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

        # 過去のパラメータを保持する変数
        self._past_variables = list()
        for variable in var_list:
            self._past_variables.append(
                self.add_variable_from_reference(
                    model_variable = variable, variable_name = "past_variable",
                    # 初期値はパラメータと同じにする
                    initial_value = variable
                )
            )
    
    def update_step(self, gradient, variable):
        # ハイパーパラメータはパラメータと同じデータ型にキャストする
        learning_rate = tf.cast(self._learning_rate, variable.dtype)
        mu = tf.cast(self._mu, variable.dtype)

        # 保持する変数を取得するためのキー
        var_key = self._var_key(variable)

        # 1イテレーション前のパラメータを取得
        past_variable = self._past_variables[self._index_dict[var_key]]
        
        # 更新前のパラメータを保持
        tmp_variable = variable.value()
        # パラメータは assign 関数で更新する
        variable.assign( variable + mu * (variable - past_variable) - learning_rate * gradient )
        # 更新前のパラメータを更新
        past_variable.assign( tmp_variable )
