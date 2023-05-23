class AdaGrad(optimizers.Optimizer):
    def __init__(self, learning_rate = 0.001, name = "AdaGrad"):
        super().__init__(name = name)

        self._learning_rate = self._build_learning_rate(learning_rate)
    
    def build(self, var_list):
        ###
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        ###
        
        self._velocity = list()
        for var in var_list:
            # 初期値は0で設定される
            self._velocity.append(
                self.add_variable_from_reference(
                    model_variable = var, variable_name = "h"
                )
            )   
    
    def update_step(self, gradient, variable):
        var_key = self._var_key(variable)
        h = self._velocity[self._index_dict[var_key]]

        h.assign(h + gradient ** 2)
        variable.assign(variable - self._learning_rate * gradient / (h ** (1 / 2) + 1e-5))
