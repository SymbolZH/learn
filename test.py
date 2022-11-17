from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, concatenate_datasets


#分词
def f(data):
    #移除<br/>
    for i in range(len(data['text'])):
        data['text'][i] = data['text'][i].replace('<br /><br />', ' ')

    data = tokenizer.batch_encode_plus(data['text'])

    return data

#加载编码器
tokenizer = AutoTokenizer.from_pretrained('gpt2')

print(tokenizer)

#编码试算
tmp = tokenizer.batch_encode_plus([
    'hide new secretions from the parental units',
    'contains no wit , only labored gags'
])


#加载数据
dataset = load_from_disk('datas/imdb')

print(dataset)
#重新切分数据集
# concatenate_datasets是将列相同的数据合并起来
dataset = concatenate_datasets(
    [dataset['train'], dataset['test'], dataset['unsupervised']])

print(dataset)

# train_test_split是将数据进行重新分割
dataset = dataset.train_test_split(test_size=0.01, seed=0)

#采样,数据量太大了跑不动
dataset['train'] = dataset['train'].shuffle(0).select(range(80000))
dataset['test'] = dataset['test'].shuffle(0).select(range(200))





dataset = dataset.map(f,
                      batched=True,
                      num_proc=4,
                      batch_size=1000,
                      remove_columns=['text', 'label'])


#过滤掉太短的句子
def f(data):
    return [sum(i) >= 25 for i in data['attention_mask']]


dataset = dataset.filter(f, batched=True, num_proc=4, batch_size=1000)


#拼合句子到统一的长度
def f(data):
    block_size = 512

    #展平数据
    input_ids = []
    for i in data['input_ids']:
        input_ids.extend(i)

    #切断数据
    data = {'input_ids': [], 'attention_mask': []}
    for i in range(len(input_ids) // block_size):
        block = input_ids[i * block_size:i * block_size + block_size]
        data['input_ids'].append(block)
        data['attention_mask'].append([1] * block_size)

    #设置labels
    data['labels'] = data['input_ids'].copy()

    return data


dataset = dataset.map(
    f,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

dataset