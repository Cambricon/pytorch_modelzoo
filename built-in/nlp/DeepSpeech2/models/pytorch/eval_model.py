import json

import torch
from torch.autograd import Variable
#from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss

import torch.nn.functional as F

import sys
### Import Data Utils ###
sys.path.append('../')

from data.bucketing_sampler import BucketingSampler, SpectrogramDatasetWithLength
from data.data_loader import AudioDataLoader, SpectrogramDataset
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns
#from params import device
try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
except ImportError:
    print("import torch_mlu failed!")

def eval_model(model, test_loader, decoder, args, iters=-1):
        start_iter = 0  # Reset start iteration for next epoch
        total_cer, total_wer = 0, 0
        model.eval()
        for i, (data) in enumerate(test_loader):  # test
            if i==iters:
                break
            inputs, targets, input_percentages, target_sizes = data

            #inputs = Variable(inputs, volatile=True)

            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            if args.device == 'gpu':
                inputs = inputs.cuda()
            elif args.device == 'mlu':
                inputs = inputs.to('mlu', non_blocking=True)

            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH
            seq_length = out.size(0)
            sizes = input_percentages.mul_(int(seq_length)).int()

            decoded_output = decoder.decode(out.data, sizes)
            target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
            wer, cer = 0, 0
            for x in range(len(target_strings)):
                wer += decoder.wer(decoded_output[x], target_strings[x]) / float(len(target_strings[x].split()))
                cer += decoder.cer(decoded_output[x], target_strings[x]) / float(len(target_strings[x]))
            total_cer += cer
            total_wer += wer

            if args.device == 'gpu':
                torch.cuda.synchronize()
            del out
        wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)
        wer *= 100
        cer *= 100

        return wer, cer
