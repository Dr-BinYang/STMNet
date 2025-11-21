from data_provider.data_loader import Dataset_Weather
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

# Dataset class mapping for different data types
data_dict = {
    'weather': Dataset_Weather
}


def data_provider(args, flag):
    """
    Data provider function that creates datasets and data loaders

    Args:
        args: Configuration arguments containing dataset parameters
        flag: Data split type ('train', 'val', 'test')

    Returns:
        tuple: (dataset, data_loader) for the specified split
    """
    # Get appropriate dataset class based on data type
    Data = data_dict[args.data]

    # Configure data loader parameters based on split type
    if flag == 'test':
        shuffle_flag = False  # No shuffling for test set
        drop_last = True  # Drop incomplete batches
        batch_size = args.batch_size  # Use configured batch size (default 1 for evaluation)
        freq = args.freq
    else:  # For training and validation
        shuffle_flag = True  # Shuffle for better training
        drop_last = True  # Drop incomplete batches
        batch_size = args.batch_size
        freq = args.freq

    # Initialize dataset
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],  # [input_sequence_length, prediction_length]
        freq=freq,
        timeenc=args.timeenc  # Time encoding method
    )

    # Print dataset information
    print(flag, len(data_set))

    # Create data loader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)

    return data_set, data_loader


if __name__ == '__main__':
    # Test data loading functionality
    parser = argparse.ArgumentParser(description='STMNet')

    # Data loader configuration
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--data', type=str, default='weather', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../dataset/weather/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--freq', type=str, default='t', help='frequency for time features')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # Parse arguments and test data provider
    args = parser.parse_args()
    print(args)
    data_set, data_loader = data_provider(args, flag='train')