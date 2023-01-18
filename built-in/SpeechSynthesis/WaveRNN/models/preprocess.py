import glob
from utils.display import *
from utils.dsp import *
from utils import hparams as hp
from multiprocessing import Pool, cpu_count
from utils.paths import Paths
import pickle
import argparse
from utils.text.recipes import ljspeech
from utils.files import get_files
from pathlib import Path


# Helper functions for argument types
def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n

parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--path', '-p', help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--extension', '-e', metavar='EXT', default='.wav', help='file extension to search for in dataset folder')
parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers, default=cpu_count()-1, help='The number of worker threads to use for preprocessing')
parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
args = parser.parse_args()

hp.configure(args.hp_file)  # Load hparams from file
if args.path is None:
    args.path = hp.wav_path

extension = args.extension
path = args.path

def convert_file(path: Path):
    y = load_wav(path)
    coarse_classes, fine_classes = split_signal(y)
    coarse_classes = np.reshape(coarse_classes, (1, -1))
    fine_classes = np.reshape(fine_classes, (1, -1))
    return coarse_classes.astype(np.int64), fine_classes.astype(np.float32)


def process_wav(path: Path):
    wav_id = path.stem
    coarse_classes, fine_classes = convert_file(path)
    np.save(paths.coarse/f'{wav_id}.npy',coarse_classes, allow_pickle=False)
    np.save(paths.fine/f'{wav_id}.npy',fine_classes, allow_pickle=False)
    return wav_id, coarse_classes.shape[0]


wav_files = get_files(path, extension)
paths = Paths(hp.data_path)

print(f'\n{len(wav_files)} {extension[1:]} files found in "{path}"\n')

if len(wav_files) == 0:

    print('Please point wav_path in hparams.py to your dataset,')
    print('or use the --path option.\n')

else:

    n_workers = max(1, args.num_workers)

    simple_table([
        ('Sample Rate', hp.sample_rate),
        ('Bit Depth', hp.bits),
        ('Mu Law', hp.mu_law),
        ('Hop Length', hp.hop_length),
        ('CPU Usage', f'{n_workers}/{cpu_count()}')
    ])

    pool = Pool(processes=n_workers)
    dataset = []

    for i, (item_id, length) in enumerate(pool.imap_unordered(process_wav, wav_files), 1):

        dataset += [(item_id, length)]
        bar = progbar(i, len(wav_files))
        message = f'{bar} {i}/{len(wav_files)} '
        stream(message)

    with open(paths.data/'dataset_WaveRNN.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    print('\n\nCompleted. Ready to run "python train.py". \n')
