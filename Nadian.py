from tensorflow.keras import optimizers

class Nadian(optimizers.Optimizer):
    """
        iterative fomula of Nadian is 
            w = w + lr * ( (1/beta - alpha) * w - 1/beta * z - beta * n_g )
            z = z + lr * ( (1/beta - alpha) * w - 1/beta * z )
            
        where, w : parameter, n_g : nesterov gradient(dE/d(w + mu * v)), v : increase in w
               lr : learning rate, mu : momentum, aplha : hyper parameter, beta : hyper parameter
    """
    def __init__(self, learning_rate = 0.001, alpha = 1.0, beta = 1.5, mu = 0.9, name = "Nadian"):
        super().__init__(name = name)

        self._learning_rate = self._build_learning_rate(learning_rate)
        self._alpha = alpha
        self._beta = beta
        self._mu = mu

    
    def build(self, var_list):
        ###
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        ###
        
        self._y = list()
        # 更新前のパラメータを保持する変数
        self._tmp_variable = list()
        # 1イテレーション前のパラメータを保持する変数
        self._past_variable = list()
        for var in var_list:
            # 初期値は0で設定される
            self._y.append(
                self.add_variable_from_reference(
                    model_variable = var, variable_name = "y"
                )
            )
            # 初期値は0で設定される
            self._tmp_variable.append(
                self.add_variable_from_reference(
                    model_variable = var, variable_name = "tmp_variable"
                )
            )
            # 初期値をパラメータと同じにする
            self._past_variable.append(
                self.add_variable_from_reference(
                    model_variable = var, variable_name = "past_variable",
                    initial_value = var
                )
            )
    
    # ネステロフの加速勾配を求めるように編集
    def compute_gradients(self, loss, var_list, tape=None):
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
        for variable in var_list:
            var_key = self._var_key(variable)
            tmp_variable = self._tmp_variable[self._index_dict[var_key]]
            past_variable = self._past_variable[self._index_dict[var_key]]

            # 今のパラメータを保持
            tmp_variable.assign(variable)
            # パラメータをネステロフの加速勾配の形に変更
            variable.assign(variable + self._mu * (variable - past_variable))

        # ネステロフの加速勾配    
        grads = tape.gradient(loss, var_list)

        # パラメータを元の形に変更
        for variable in var_list:
            var_key = self._var_key(variable)
            tmp_variable = self._tmp_variable[self._index_dict[var_key]]

            variable.assign(tmp_variable)

        return list(zip(grads, var_list))
    
    def update_step(self, gradient, variable):
        var_key = self._var_key(variable)
        y = self._y[self._index_dict[var_key]]
        tmp_variable = self._tmp_variable[self._index_dict[var_key]]
        past_variable = self._past_variable[self._index_dict[var_key]]

        # 現在のパラメータを保持
        past_variable.assign(variable)
        # パラメータを保持
        tmp_variable.assign(variable)
        # zを更新
        variable.assign(variable + self._learning_rate * ( (1 / self._beta - self._alpha) *  variable - 1 / self._beta * y - self._beta * gradient) )
        y.assign(y + self._learning_rate * ( (1 / self._beta - self._alpha) *  variable - 1 / self._beta * y ) )
