class CompileParams:
    def __init__(self, loss, metrics, optimizer):
        self.loss, self.metrics, self.optimizer = loss, metrics, optimizer


def compile_model(model, compile_params: CompileParams):
    model.compile(
        loss=compile_params.loss,
        metrics=compile_params.metrics,
        optimizer=compile_params.optimizer
    )
