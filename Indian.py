from tensorflow.keras import optimizers

class Indian(optimizers.Optimizer):
    """
        iterative fomula of Indian is 
            w = w + lr * ( (1/beta - alpha) * w - 1/beta * z - beta * g )
            z = z + lr * ( (1/beta - alpha) * w - 1/beta * z )
            
        where, w : parameter, g : gradient, 
               lr : learning rate, aplha : hyper parameter, beta : hyper parameter
    """
    def __init__(self, learning_rate = 0.001, alpha = 1.0, beta = 1.5, name = "Indian"):
        super().__init__(name = name)

        self._learning_rate = self._build_learning_rate(learning_rate)
        self._alpha = alpha
        self._beta = beta
    
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
        for var in var_list:
            # 初期値は0で設定される
            self._y.append(
                self.add_variable_from_reference(
                    model_variable = var, variable_name = "y"
                )
            )
            self._tmp_variable.append(
                self.add_variable_from_reference(
                    model_variable = var, variable_name = "tmp_variable"
                )
            )
    
    def update_step(self, gradient, variable):
        var_key = self._var_key(variable)
        y = self._y[self._index_dict[var_key]]
        tmp_variable = self._tmp_variable[self._index_dict[var_key]]

        # パラメータを保持
        tmp_variable.assign(variable)
        # zを更新
        variable.assign(variable + self._learning_rate * ( (1 / self._beta - self._alpha) *  variable - 1 / self._beta * y - self._beta * gradient) )
        y.assign(y + self._learning_rate * ( (1 / self._beta - self._alpha) *  tmp_variable - 1 / self._beta * y ) )
