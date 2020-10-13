##############for dataset############
HEIGHT = 224
WIDTH = 224
DATAPATH = "/defaultShare/share/wujl/83/ddb_classall"
THREAD = 64
BATCHSIZE=256
LOADERNAME="zlc_multi_balance"
#############for train ##############
GPUS = '0,1,2,3'
LOADING = False 
LOADPATH = ''
MODELNAME = "resnext101_32x8d"
NUMCLASS = 19
INITLR = 0.01
LOGPATH = '/defaultShare/share/wujl/83/master_models/resnext101_19_mixcut/log'
WD = 1e-4
WARM = 4
LRTAG = 'MultiStep'
EPOCHS=80
MIXUP = True
Cut = True
############# for save ###############
SAVEFREQ = 2
MODELSAVEPATH="/defaultShare/share/wujl/83/master_models/resnext101_19mixcutv2"
############# for eval ###############
MAPLIST = ['badcase', 'feigang', 'hanzha', 'laji00', 'laji01', 'laji02', 'laji03', 'luanfang', 'zangwu',
           'zhufei', 'xianshu_luan', 'xianshu_zhengqi', 'xiaojian_luan', 'xiaojian_zhengqi', 'pip_luan', 'pip_zhengqi', 'zhixiang',
           'bancai_luan', 'bancai_zhengqi']

