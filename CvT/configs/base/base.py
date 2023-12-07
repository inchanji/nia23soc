import os

class CFG():
    def __init__(self):
        self.projectName     = 'nia23soc-image-classification'
        self.data_root       =  "./dataset" 
        self.model_arch      = 'microsoft/cvt-13'
        self.imgsize         = 512

        self.epochs          = 100
        self.patience        = 5
        self.train           = True        

        self.n_fold          = 1

        self.dataset         = None

        self.seed           = 42
        self.loss           = None

        self.train_bs        = 4
        self.valid_bs        = 8
        self.num_workers     = 4
        self.optimizer       = 'adam'
        self.apex            = True

        self.lr              = 1e-4
        self.min_lr          = 1e-6
        self.weight_decay    = 1e-6
        self.weight_lm       = False
        self.accum_iter      = 1 # suppoprt to do batch accumulation for backprop with effectively larger batch size
        self.verbose_step    = 1

        self.num_classes      = 10
        self.include_normal   = True
        self.debug          = False

        # DDP related
        self.world_size      = 1
        self.rank            = -1

    def setExpName(self):
        return f"{self.model_arch}_{self.imgsize}px"

    def finalize(self):
        self.img_h           = self.imgsize      
        self.img_w           = self.imgsize   

        if self.scheduler == 'CosineAnnealingLR':
            self.T_max       = self.epochs // 5# CosineAnnealingLR
        elif self.scheduler == 'ReduceLROnPlateau':
            self.factor      = 0.2 # ReduceLROnPlateau
            self.eps         = 1e-6 # ReduceLROnPlateau
        elif self.scheduler == 'CosineAnnealingWarmRestarts':
            self.T_0         =  self.epochs // 4 # CosineAnnealingWarmRestarts
        elif self.scheduler == 'DecayingOscillation' or self.scheduler == 'DecayingOscillation_v2':
            self.osc_t       = self.epochs
            self.tau         = 0.7*0.693147

        if self.include_normal:
            self.num_classes += 1

        self.expName         = self.setExpName()



