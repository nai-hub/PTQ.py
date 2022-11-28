_base_ = [
    '../_base_/models/resnet18_cifar.py', 
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', 
    '../_base_/default_runtime.py'
]

"""-----------------------------------common-------------------------------"""

target_type = 'classification'
workflow = [('train', 1),('val',1)]
cudnn_benchmark = True
evaluation = dict(metric='accuracy', metric_options={'topk': (1, 5)})

"""-----------------------------------for pruning--------------------------"""
# Reinforcement Learning
action_space = 1
doTraining = True

lr_a = 0.1               # learning rate for actor
lr_c = 0.1               # 
bsize = 32               # minibatch size
tau = 0.01               # moving average for target network
discount = 1.0
epsilon = 50000          # linear decay of exploration policy
init_delta = 0.5         # initial variance of truncated normal distribution
delta_decay = 0.95       # delta decay during exploration

rmsize = 100             # memory size for each layer
window_length = 1
warmup = 100             # time without training but only filling the replay memory

MaxEpisodes = 5000 
MaxStepsPerEpisode = 5 
StopTrainingCriteria = 'AverageReward'
StopIterationValue = 1
StopTrainingValue = 1.0  # top-1 acc_error 1.0%
compress_ratio = 0.30    # 30%

# KD fine-tune for classification
distill_trainer = dict(
    type = 'ClassificationDistillTrainer',
    kd_config=dict(
        temperature = 2,
        lamda = 0.9)
    )
distill_optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
distill_lr_config = dict(policy='step', step=[30])
distill_runner = dict(type='EpochBasedRunner', max_epochs=1) 

"""-----------------------------------for quantization--------------------------"""
model_size_limit_ratio = 0.5
quant_type = 'mix'

# QAT fine-tune for quantization
bitwidth=[2,2,2,2,2,2,2,1,2,2,2,2,1,2,2,2,1,1,1,1,1]
qat_optimizer = dict(type='SGD', lr=0.00001, momentum=0.9, weight_decay=0.0005)
qat_lr_config = dict(policy='step', step=[30])
qat_runner = dict(type='EpochBasedRunner', max_epochs=3) 
selfdistill_model = False





