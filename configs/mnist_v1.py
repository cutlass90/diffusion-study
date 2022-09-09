class Config:
    def __init__(self):
        self.lr = 2e-4
        self.device = 'cuda:0'
        self.checkpoint_dir = 'checkpoints/mnist_v2'
        self.log_freq = 50
        self.save_freq = 300
        self.load_path = 'checkpoints/mnist_v2/weights/latest.pth'
        # self.load_path = ''
        self.batch_size = 128
        self.filters = 64
        self.n_epoch = 100
        self.diffusion_steps = 1000

    def __str__(self):
        stringa = ''
        for k, v in self.__dict__.items():
            stringa += f'{k}={v}' + '\n'
        return stringa



opt = Config()