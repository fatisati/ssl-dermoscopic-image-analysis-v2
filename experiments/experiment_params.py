import pandas as pd


class ExperimentParams:
    def __init__(self, data, bs, img_size, aug_func,
                 batchnorm, dropout, attention, hidden_dim,
                 model, loss, opt, epochs):
        self.data, self.bs, self.img_size, self.aug_func = data, bs, img_size, aug_func
        self.batchnorm, self.dropout, self.attention, self.hidden_dim = batchnorm, dropout, attention, hidden_dim
        self.model, self.loss, self.opt, self.epochs = model, loss, opt, epochs

    def save_params(self, path):
        df = pd.DataFrame([self.__dict__])
        df.to_csv(path + 'params.csv')
