import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

i = 0
current_lr = 2
train_loss = 3
eval_loss = 4
ptrain = 12
peval = 11

fcs.print_info(
    Epoch=[i+1],
    LR=[current_lr],
    TRAIN__slash__VALID=[str(train_loss)+'/'+str(eval_loss)],
    TRIAN__slash__VALID__percent__=[str(ptrain)+'%/'+str(peval)+'%']
)