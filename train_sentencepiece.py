import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="en.txt,fr.txt",
    model_prefix="en_fr",
    vocab_size=16000,
    model_type="bpe",
    input_sentence_size = 5000000,
    shuffle_input_sentence = True,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)
