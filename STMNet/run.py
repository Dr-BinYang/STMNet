import argparse
import torch
import random
import numpy as np
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

# Define argument parser for STMNet model configuration
parser = argparse.ArgumentParser(description='STMNet')

# Data loader configuration
parser.add_argument('--data', type=str, default='weather', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/weather/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
parser.add_argument('--freq', type=str, default='10min',
                    help='frequency for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--timeenc', type=int, default=0, help='time encoding method, 0 or 1')

# Model checkpoint configuration
parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='location of model checkpoints')

# Forecasting task configuration
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=192, help='prediction sequence length: 96, 192, 336, 720')

# Model architecture parameters
parser.add_argument('--top_k', type=int, default=5, help='top_k value for attention')
parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
parser.add_argument('--e_layers', type=int, default=2, help='number of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='number of decoder layers')
parser.add_argument('--d_ff', type=int, default=1024, help='dimension of feed forward network')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--activation', type=str, default='gelu', help='activation function')
parser.add_argument('--moving_avg', type=int, default=15, help='window size for moving average')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method for series decomposition, supports: moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalization; 1 for True, 0 for False')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='number of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, supports: avg, max, conv')
parser.add_argument('--c_in', type=int, default=21, help='input feature size')
parser.add_argument('--c_out', type=int, default=21, help='output feature size')

# MTM (Multi-scale Temporal Memory) module configuration
parser.add_argument('--MTM', type=bool, default=True, help='use MTM module')
parser.add_argument('--MTM_factor', type=int, default=4, help='decomposition factor for MTM')

# Padding strategy
parser.add_argument('--padStrategy', type=str, default='zeros', help='padding strategy: repeat, zeros, mean')

# Spatio-Temporal Fusion configuration
parser.add_argument('--STFus', type=bool, default=True, help='use Spatio-Temporal Fusion')
parser.add_argument('--STFusTempor', type=str, default='Attention', help='Temporal Module type: GRU, Attention')

# Training optimization parameters
parser.add_argument('--train_epochs', type=int, default=3000, help='training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--patience', type=int, default=15, help='scheduler patience')
parser.add_argument('--early_stop_patience', type=int, default=40, help='early stopping patience')

# GPU configuration
parser.add_argument('--use_gpu', type=bool, default=True, help='use GPU')

# Visualization and plotting configuration
parser.add_argument('--plot_result', type=bool, default=False, help='plot prediction results')
parser.add_argument('--plot_hierarchy_prediction', type=bool, default=False,
                    help='plot hierarchical prediction evolution')
parser.add_argument('--plot_attn_weights', type=bool, default=False, help='plot attention weights')
parser.add_argument('--plot_loss_curve', type=bool, default=False, help='plot training loss curve')

if __name__ == '__main__':
    # Parse command line arguments
    args = parser.parse_args()

    # Configure GPU usage based on availability
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # Uncomment below line to force CPU usage
    # args.use_gpu = False

    print('Args in experiment:')
    print(args)

    # Create experiment setting description
    setting = 'data:{} seqlen:{} predlen:{} d_model:{} n_heads:{} e_layers:{} d_layers:{} d_ff:{} '.format(
        args.data,
        args.seq_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff
    )

    # Initialize experiment
    exp = Exp_Long_Term_Forecast(args)

    # Start training
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train()