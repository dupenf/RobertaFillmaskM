import torch

torch.cuda.is_available()
from transformers import RobertaConfig

is_train = False

if is_train:
    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1
    )

    from transformers import RobertaTokenizerFast, RobertaForMaskedLM

    tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBerto")
    model = RobertaForMaskedLM(config=config)
    print(model.parameters())
    #####################################################
    from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="./datasets/eng_news-typical_2016_1M-sentences.txt",
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    from transformers import Trainer, TrainingArguments

    training_arg = TrainingArguments(
        output_dir="./models",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_gpu_eval_batch_size=640,
        save_steps=10_000,
        save_total_limit=2,
        use_cpu=False,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_arg,
        data_collator=data_collator,
        train_dataset=dataset,
        # prediction_loss_only=True
    )

    trainer.train()
    trainer.save_model("./models")

else:
    from transformers import pipeline
    print("------------------------fill mask--------------------------")
    fill_mask = pipeline(
        task="fill-mask",
        model="./models/checkpoint-10000",
        tokenizer="./models/checkpoint-10000"
    )
    pred = fill_mask("Entrant’s <mask> Personal <mask>.")
    # 两个mask则会给一个list
    for p in pred:
        for x in p:
            print("->")
            print(x)
            # 这是一个翻译模型吗？
