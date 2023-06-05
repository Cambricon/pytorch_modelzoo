import os
import sys
import time
import logging
import torch
import torch.multiprocessing as mp
import argparse
from tqdm.auto import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from util import load_data, T5PegasusTokenizer, KeyDataset, create_data, default_collate, compute_rouges,set_seed
from transformers import MT5ForConditionalGeneration
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../tools/utils/")
from metric import MetricCollector
try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
    ct.set_memory_strategy(True)
except ImportError:
    print("import torch_mlu failed!")



# 定义目标精度
target_precision = {
    "rouge-l": 0.5443,
    "rouge-1": 0.5864,
    "rouge-2": 0.4421
}

# 声明梯度缩放器
scaler = GradScaler()


def evaluate(args):
    device_id = 0
    device = torch.device("mlu:{}".format(device_id))
    ct.set_device(device)
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
    dev_data = prepare_data(args, args.dev_data, tokenizer,
                            rank=-1, world_size=-1, term='dev')
    logger = logging.getLogger("evaluate")

    model_path = os.path.join(args.saved_model_dir,args.saved_model_name)
    if not os.path.exists(model_path):
        raise Exception("{} not exists".format(model_path))
    model = torch.load(model_path, map_location='cpu')
    model = model.to(device)
    model.eval()
    gens = []
    summaries = []
    metric_collector = MetricCollector(enable_only_avglog=True,
                                       record_elapsed_time=True,
                                       record_hardware_time=True if args.device == 'MLU' else False)
    metric_collector.place()
    for index, feature in tqdm(enumerate(dev_data)):
        if (index == args.eval_iterations):
            logger.info(
                'The program iteration runs out. valid_iterations: %d' % args.eval_iterations)
            break
        title = feature['title']
        content = {k: v.to(device)
                   for k, v in feature.items() if k != 'title'}
        gen = model.generate(max_length=args.max_len_generate,
                             eos_token_id=tokenizer.sep_token_id,
                             decoder_start_token_id=tokenizer.cls_token_id,
                             **content)
        gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
        gen = [item.replace(' ', '') for item in gen]
        gens.extend(gen)
        summaries.extend(title)
        metric_collector.record()
        metric_collector.place()
    scores = compute_rouges(gens, summaries)
    metric_collector.insert_metrics(net="mt5", accuracy=scores)
    metric_collector.dump()

def main(args):
    if args.device == "CPU" and args.distributed:
        print("The CPU device platform does not support distributed operation.")
        return
    if args.num_device < 1:
        print("Error: num_device must >= 1")
        return
    if args.num_device == 1 and args.distributed:
        print("Error: when num_device == 1, distributed should be False")
        return
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.saved_model_dir):
        os.makedirs(args.saved_model_dir)
    if args.seed is not None:
        set_seed(args.seed)

    if args.device == "MLU":
        ndev = ct.device_count()
    if args.device == "GPU":
        ndev = torch.cuda.device_count()
    if ndev < args.num_device:
        print("Error: mt5 demo expect {} device, but only {} found".format(
            args.num_device, ndev))
        return
    
    if args.mode == 'evaluation':
        evaluate(args)
        return

    # 并行训练模型
    # spawn_processes(args.num_device, train_model, tokenizer, args)
    start = time.time()
    if args.distributed:
        if (sys.version_info[0] < 3):
            print("Error: does not support python2, use python3 instead")
            sys.exit(1)
        else:
            if not os.getenv('MASTER_ADDR'):
                os.environ['MASTER_ADDR'] = args.master_addr
            if not os.getenv('MASTER_PORT'):
                os.environ['MASTER_PORT'] = args.master_port
            mp.spawn(train, nprocs=args.num_device, args=(args.num_device, args), join=True)
    else:
        train(0, ndev, args)
    end = time.time()
    print("Using Time: " + str(end-start))


def prepare_data(args, data_path, tokenizer, rank, world_size, term='train'):
    """准备batch数据
    """
    data = load_data(data_path)
    data = create_data(data, tokenizer, args.max_len, term)
    data = KeyDataset(data)
    if term == "dev" or (term=="train" and not args.distributed):
        data = DataLoader(data, batch_size=args.batch_size,
                          collate_fn=default_collate)
    else:
        sampler = DistributedSampler(data, rank=rank, num_replicas=world_size)
        data = DataLoader(data, batch_size=args.batch_size,
                          collate_fn=default_collate, sampler=sampler)
    return data


def train(rank, world_size, args):
    train_start = time.time()
    logger = logging.getLogger("rank{}".format(rank))
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(
        args.log_path, "rank{}.out.log".format(rank)))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[PerfLog] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


    stop_training = 0
    device_id = rank
    if args.device == "MLU":
        device = torch.device("mlu:{}".format(device_id))
        ct.set_device(device)
    if args.device == "GPU":
        device = torch.device("cuda:{}".format(device_id))
        torch.cuda.set_device(device)
        # distributed training env setting up
    if args.distributed:
        dist.init_process_group(
            backend='cncl' if args.device == "MLU" else 'nccl', rank=rank, world_size=world_size)
    # 加载T5PegasusTokenizer分词器
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)

    if args.resume_model != "" and os.path.exists(args.resume_model):
        model = torch.load(args.resume_model, map_location='cpu').to(device)
    else:
        model = MT5ForConditionalGeneration.from_pretrained(
        args.pretrain_model).to(device)

    if args.distributed:
        model = DDP(model, device_ids=[rank])

    train_data = prepare_data(
        args, args.train_data, tokenizer, rank=rank, world_size=world_size, term='train',)
    dev_data = prepare_data(args, args.dev_data, tokenizer,
                            rank=rank, world_size=world_size, term='dev')

    if rank == 0:
        if not os.path.exists(args.saved_model_dir):
            os.mkdir(args.saved_model_dir)

    best = 0
    best_epoch = 0
    best_scores = 0
    adam = torch.optim.Adam(model.parameters(), lr=args.lr)
    global_steps = 0

    bare_train_time = 0
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    if args.device == "GPU":
        torch.cuda.synchronize()

    for epoch in range(args.num_epoch):
        model.train()
        metric_collector = MetricCollector(
            enable_only_benchmark=True,
            record_elapsed_time=True,
            record_hardware_time=True if args.device == 'MLU' else False)
        epoch_start = time.time()
        metric_collector.place()
        for i, cur in enumerate(train_data):
            if (i == args.train_iterations):
                logger.info('The program iteration runs out. train_iterations: %d' % args.train_iterations)
                break
            titles = cur['title']
            cur = {k: v.to(device) for k, v in cur.items() if k != "title"}
            with autocast(enabled=args.amp):
                output = model(**cur)[0]
                mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
                output = output[:, :-1]
                prob = output.reshape((-1, output.size(-1)))[mask]
                labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
                if args.device == "GPU":
                    labels = labels.to(torch.long)
                loss = loss_fct(prob, labels)
                gens = []
                batch_size = output.shape[0]
                for j in range(batch_size):
                    predict_label = torch.argmax(output[j], axis=1)
                    gen = tokenizer.batch_decode(
                        predict_label[cur['decoder_attention_mask'][j][1:].bool()], skip_special_tokens=True)
                    gen = [item.replace(' ', '') for item in gen]
                    gens.append(gen)
                scores = compute_rouges(gens, titles)  # right or wrong
                scores = {
                    k: "{:.4f}".format(v) for k, v in scores.items()
                }
                global_steps += 1
                logger_item = {}
                logger_item['event'] = "STEP_END"
                logger_item["value"] = {
                    "epoch": epoch,
                    "global_steps": global_steps,
                    "loss": "{:.4f}".format(loss.item()),
                    "acc": scores,
                    "num_trained_samples": global_steps * args.batch_size,
                    "learning_rate": args.lr
                }
                logger.info(logger_item)
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(adam)
                scaler.update()
            else:
                loss.backward()
                adam.step()
            adam.zero_grad()
            metric_collector.record()
            metric_collector.place()
        epoch_end = time.time()

        metrics = metric_collector.get_metrics()
        if 'batch_time_avg' in metrics:
            metric_collector.insert_metrics(
                throughput=args.batch_size / metrics['batch_time_avg'] * args.num_device)
        metric_collector.insert_metrics(
            net="mt5",
            batch_size=args.batch_size,
            precision="amp" if args.amp else "fp32",
            cards=args.num_device,
            DPF_mode="ddp " if args.distributed == True else "single")
        if ((args.distributed == False) or (rank == 0)):
            metric_collector.dump()

        # 验证
        if rank == 0:
            model.eval()
            gens = []
            gen_ids = []
            summaries = []
            eval_start = time.time()
            for index, feature in tqdm(enumerate(dev_data)):
                if (index== args.valid_iterations):
                    logger.info('The program iteration runs out. valid_iterations: %d' % args.valid_iterations)
                    break
                title = feature['title']
                content = {k: v.to(device)
                           for k, v in feature.items() if k != 'title'}
                if args.distributed:
                    gen = model.module.generate(max_length=args.max_len_generate,
                                                eos_token_id=tokenizer.sep_token_id,
                                                decoder_start_token_id=tokenizer.cls_token_id,
                                                **content)
                else:
                    gen = model.generate(max_length=args.max_len_generate,
                                                eos_token_id=tokenizer.sep_token_id,
                                                decoder_start_token_id=tokenizer.cls_token_id,
                                                **content)
                gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
                gen = [item.replace(' ', '') for item in gen]
                gens.extend(gen)
                summaries.extend(title)
            scores = compute_rouges(gens, summaries)
            print("Validation score: {}".format(scores))
            rouge_l = scores['rouge-l']
            eval_end = time.time()
            logger_item = {}
            logger_item['event'] = "EVALUATE_END"
            logger_item["value"] = {
                "epoch": epoch,
                "global_steps": i,
                "eval_mlm_accuracy": {
                    k: "{:.4f}".format(v) for k, v in scores.items()
                },
                "eval_time": eval_end - eval_start,
                "epoch_time":  epoch_end - epoch_start
            }
            bare_train_time += epoch_end - epoch_start
            logger.info(logger_item)
            if rouge_l > best:
                best = rouge_l
                best_epoch = epoch
                best_scores = scores
                if args.distributed:
                    torch.save(model.module, os.path.join(
                        args.saved_model_dir, args.saved_model_name))
                else:
                    torch.save(model, os.path.join(
                        args.saved_model_dir, args.saved_model_name))
            if args.early_stop and best_scores['rouge-l'] >= target_precision['rouge-l'] and \
                    best_scores['rouge-1'] >= target_precision['rouge-1'] and \
                    best_scores['rouge-2'] >= target_precision['rouge-2']:
                stop_training = 1

        if args.early_stop and args.distributed:
            if rank == 0:
                dist.broadcast(torch.tensor(stop_training).to(device), rank)
            else:
                stop_signal_tensor = torch.zeros(1, dtype=torch.int32).to(device)
                dist.broadcast(stop_signal_tensor, 0)
                stop_training = stop_signal_tensor.cpu().item()

        if args.early_stop and stop_training:
            logger.info("target precison got, early stop")
            break

    if rank == 0:
        train_end = time.time()
        num_trained_samples = global_steps * args.batch_size * world_size
        logger_item = {}
        logger_item['event'] = "TRAIN_END"
        logger_item["value"] = {
            "accuracy": best_scores,
            "epoch": best_epoch,
            "train_time": train_end - train_start,
            "num_trained_samples": num_trained_samples,
            "samples/sec":  num_trained_samples / bare_train_time
        }
        logger.info(logger_item)


if __name__ == '__main__':

    # 初始化参数
    parser = argparse.ArgumentParser(description='mt5')
    parser.add_argument('--seed', default=66, type=int, help='random seed.')
    parser.add_argument('--log-path', default='logs', type=str, help='training log path.')
    parser.add_argument( '--train_data', default='./data/summarization_csl_train.tsv')
    parser.add_argument( '--dev_data', default='./data/summarization_csl_dev.tsv')
    parser.add_argument('--pretrain_model', default='./t5_pegasus_pretrain')
    parser.add_argument('--saved_model_dir', default='./saved_model')
    parser.add_argument('--saved_model_name', default='summary_model')
    parser.add_argument('--amp', action="store_true", default=False)
    parser.add_argument('--num_epoch', default=4, type=int, help='number of epoch')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--lr', default=2e-4, type=float,help='learning rate')
    parser.add_argument('--distributed', action='store_true', help='distributed training.')
    parser.add_argument('--max_len', default=512, type=int,help='max length of inputs')
    parser.add_argument('--max_len_generate', default=40, type=int,help='max length of outputs')
    parser.add_argument('--train_iterations', default=-1, type=int, help="Number of training iterations.")
    parser.add_argument('--valid_iterations', default=-1, type=int, help="Number of validation iterations.")
    parser.add_argument('--eval_iterations', default=-1, type=int, help="Number of evaluation iterations.")
    parser.add_argument('--num_device', default=1, type=int, help='number of device to use')
    parser.add_argument('--device', default='MLU', type=str, help='set the type of hardware used for training.')
    parser.add_argument('--master-addr', default='127.0.0.1', type=str, help='ddp address.')
    parser.add_argument('--master-port', default='29502', type=str, help='ddp address port.')
    parser.add_argument('--mode', type=str, default='training', choices=['training', 'evaluation'])
    parser.add_argument('--early_stop', action="store_true", default=False,help="if True, will early stop when valid accuracy is better than target accurary")
    parser.add_argument('--resume_model', type=str, default="", help="resume model path")



    args = parser.parse_args()

    main(args)
