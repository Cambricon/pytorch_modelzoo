import torch
from torch.utils.data import DataLoader
import csv
import argparse
from tqdm.auto import tqdm
from util import load_data, T5PegasusTokenizer, KeyDataset, default_collate, set_seed

def create_data(data, tokenizer, max_len):
    """
    调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    :param data: 未经tokenize的数据
    :param tokenizer: tokenizer对象
    :param max_len: 最大长度
    :return: 编码后的数据集
    """
    ret, flag, title = [], True, None
    for content in data:
        if type(content) == tuple:
            title, content = content
        text_ids = tokenizer.encode(content, max_length=max_len,
                                    truncation='only_first')

        if flag:
            flag = False
            print(content)

        features = {'input_ids': text_ids,
                    'attention_mask': [1] * len(text_ids),
                    'raw_data': content}
        if title:
            features['title'] = title
        ret.append(features)
    return ret


def prepare_data(args, tokenizer):
    """
    准备batch数据
    :param args: 命令行参数
    :param tokenizer: tokenizer对象
    :return: 测试数据DataLoader
    """
    test_data = load_data(args.test_data)
    test_data = create_data(test_data, tokenizer, args.max_len)
    test_data = KeyDataset(test_data)
    test_data = DataLoader(
        test_data, batch_size=args.batch_size, collate_fn=default_collate)
    return test_data


def generate(args):
    """
    对测试集进行预测并写入result_file中
    :param args: 命令行参数
    """
    if args.seed is not None:
        set_seed(args.seed)

    gens, summaries = [], []

    # 加载分词器
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)

    # 准备测试数据
    test_data = prepare_data(args, tokenizer)

    # 加载模型
    model = torch.load(args.model, map_location='cpu')
    if args.device == "MLU":
        device = "mlu:0"
    elif args.device == "GPU":
        device = "cuda:0"
    elif args.device == "GPU":
        device = "cpu"
    else:
        print("Error: device must be one of MLU, GPU, CPU, but got MLU, GPU, CPU, but got {}".format(
            args.device))
        exit(1)

    # 将模型移动到MLU或GPU上
    model = model.to(device)

    with open(args.result_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        model.eval()
        for index,feature in tqdm(enumerate(test_data)):
            if (index == args.eval_iterations):
                    print('The program iteration runs out. valid_iterations: %d' % args.eval_iterations)
                    break
            raw_data = feature['raw_data']
            content = {k: v.to(device) for k, v in feature.items()
                       if k not in ['raw_data', 'title']}
            gen = model.generate(max_length=args.max_len_generate,
                                 eos_token_id=tokenizer.sep_token_id,
                                 decoder_start_token_id=tokenizer.cls_token_id,
                                 **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            # 去掉生成文本中的空格并写入文件
            gen = [item.replace(' ', '') for item in gen]
            writer.writerows(zip(gen, raw_data))
            gens.extend(gen)
            if 'title' in feature:
                summaries.extend(feature['title'])
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
    parser.add_argument( '--test_data', default='./data/summarization_csl_predict.tsv')
    parser.add_argument('--result_file', default='./predict_result.tsv')
    parser.add_argument('--pretrain_model', default='./t5_pegasus_pretrain')
    parser.add_argument('--model', default='./saved_model/summary_model')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--max_len', default=512, type=int, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=40, type=int, help='max length of generated text')
    parser.add_argument('--eval_iterations', default=-1, type=int, help='Number of validation evaluate')
    parser.add_argument('--device', default='MLU', type=str, help='set the type of hardware used for evaluation.')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    args = parser.parse_args()
    if args.device == "MLU":
        import torch_mlu
        import torch_mlu.core.mlu_model as ct

    generate(args)
