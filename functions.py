def get_loader(text_lens=100):
    import torch
    import random
    from transformers import BertTokenizer
    from datasets import Dataset

    tokenizer = BertTokenizer(vocab_file='tokenizer/vocab.txt',
                              model_max_length=512)

    def f():
        for _ in range(2000):
            label = random.randint(0, 9)
            text = ' '.join(str(label) * text_lens)
            yield {'text': text, 'label': label}

    dataset = Dataset.from_generator(f)

    def f(data):
        text = [i['text'] for i in data]
        label = [i['label'] for i in data]

        data = tokenizer(text,
                         padding=True,
                         truncation=True,
                         max_length=512,
                         return_tensors='pt')

        data['labels'] = torch.LongTensor(label)

        return data

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=32,
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=f)

    return tokenizer, dataset, loader


def get_model(num_hidden_layers=32):
    import torch
    from transformers import BertConfig, BertForSequenceClassification
    from transformers.optimization import get_scheduler

    config = BertConfig(num_labels=10, num_hidden_layers=num_hidden_layers)
    model = BertForSequenceClassification(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = get_scheduler(name='cosine',
                              num_warmup_steps=0,
                              num_training_steps=50,
                              optimizer=optimizer)

    return model, optimizer, scheduler