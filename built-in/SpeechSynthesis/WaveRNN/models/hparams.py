
# CONFIG -----------------------------------------------------------------------------------------------------------#

# Here are the input and output data paths (Note: you can override wav_path in preprocess.py)
wav_path = './dataset/LJSpeech-1.1/wavs/'
data_path = './dataset/data/'

sample_rate = 22050
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 275                    # 12.5ms - in line with Tacotron 2 paper
win_length = 1100                   # 50ms - same reason as above
fmin = 40
min_level_db = -100
ref_level_db = 20
bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode below
peak_norm = False                   # Normalise to the peak of each wav file


# WAVERNN_deepmind -----------------------------------------------------------------------------------------------------#
dpm_lr = 1e-4
dpm_batch_size = 32
dpm_hidden_size = 896
dpm_quantisation = 256
dpm_seq_len = 960
dpm_test_rate = 0.2



