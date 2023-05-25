import tensorflow as tf
from tensorflow.keras import optimizers

class Nadian(optimizers.Optimizer):
    def __init__(self, learning_rate = 0.01, mu = 0.9, alpha = 0.5, beta = 0.1, name = "Nadian"):
        super().__init__(name = name)
        # 学習率の設定には _build_learning_rate 関数を用いる
        self._learning_rate = self._build_learning_rate(learning_rate)
        self._mu = mu
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
        # 1イテレーション前のパラメータを保持する変数
        self._past_variables = list()
        for variable in var_list:
            self._y.append(
                self.add_variable_from_reference(
                    model_variable = variable, variable_name = "y",
                    # 初期値は0
                    initial_value = tf.zeros(shape = variable.shape)
                )
            )
            self._past_variables.append(
                self.add_variable_from_reference(
                    model_variable = variable, variable_name = "past_variable",
                    # 初期値はパラメータと同じ
                    initial_value = variable
                )
            )
    
    # ネステロフの加速勾配を求めるように編集
    def compute_gradients(self, loss, var_list, tape=None):
        # build関数を呼び出さないと保持する変数が定義されないので, ここで呼び出し
        self.build(var_list)

        if not callable(loss) and tape is None:
            raise ValueError(
                "`tape` is required when a `Tensor` loss is passed. "
                f"Received: loss={loss}, tape={tape}."
            )
        if tape is None:
            tape = tf.GradientTape()
        if callable(loss):
            with tape:
                if not callable(var_list):
                    tape.watch(var_list)
                loss = loss()
                if callable(var_list):
                    var_list = var_list()
        
        # ネステロフの加速勾配の形にパラメータを変更
        tmp_variables = list()
        for variable in var_list:
            var_key = self._var_key(variable)
            past_variable = self._past_variables[self._index_dict[var_key]]
            mu = tf.cast(self._mu, variable.dtype)

            # 今のパラメータを保持
            tmp_variables.append( variable.value() )
            # パラメータをネステロフの加速勾配の形に変更
            variable.assign(variable + mu * (variable - past_variable))

        # ネステロフの加速勾配    
        grads = tape.gradient(loss, var_list)

        # パラメータを元の形に変更
        for variable, tmp_variable in zip(var_list, tmp_variables):
            variable.assign(tmp_variable)

        return list(zip(grads, var_list))
    
    def update_step(self, gradient, variable):
        # ハイパーパラメータはパラメータと同じデータ型にキャストする
        learning_rate = tf.cast(self._learning_rate, variable.dtype)
        alpha = tf.cast(self._alpha, variable.dtype)
        beta = tf.cast(self._beta, variable.dtype)

        # 保持する変数を取得するためのキー
        var_key = self._var_key(variable)

        # yを取得
        y = self._y[self._index_dict[var_key]]
        past_variable = self._past_variables[self._index_dict[var_key]]
        
        # 更新前のパラメータを保持
        tmp_variable = variable.value()
        # パラメータは assign 関数で更新する
        variable.assign( variable + learning_rate * ( ((1/beta) - alpha) * variable - (1/beta) * y - beta * gradient ) )
        # yを更新
        y.assign( y + learning_rate * ( ((1/beta) - alpha) * tmp_variable - (1/beta) * y ) )
        # 1イテレーション前のパラメータを更新
        past_variable.assign( tmp_variable )
