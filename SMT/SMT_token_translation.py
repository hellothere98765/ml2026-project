import sentencepiece as spm
import torch 
import os 
import pandas as pd


def invert_index(dataset_path, tokenizer_path, output_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    csv_reader = pd.read_csv(
        dataset_path,
        chunksize=10000,
        header=None,
        names=["en", "fr"]
    )

    for chunk_id, chunk in enumerate(csv_reader):
        for eng, frc in zip(chunk["en"].values, chunk["fr"].values):

            if pd.isna(eng) or pd.isna(frc):
                continue
            
            eng_ids = set(sp.encode(str(eng), out_type=int))
            frc_ids = set(sp.encode(str(frc), out_type=int))

            for tok in eng_ids:
                index_en[tok].append(sentence_id)

            for tok in frc_ids:
                index_fr[tok].append(sentence_id)
            
            sentence_id += 1

    index_en = {
        tok: torch.tensor(ids, dtype=torch.int32)
        for tok, ids in index_en.items()
    }

    index_fr = {
        tok: torch.tensor(ids, dtype=torch.int32)
        for tok, ids in index_fr.items()
    }

    torch.save({
        "index_en": index_en,
        "index_fr": index_fr,
        "num_sentences": sentence_id
    }, output_path)

    print("Saved indices!")

def pretokenize(dataset_path, tokenizer_path, output_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    eng_data = []
    frc_data = []

    total = 0

    csv_reader = pd.read_csv(
        dataset_path,
        chunksize=10000,
        header=None,
        names=["en", "fr"]
    )

    for chunk in csv_reader:
        for _, row in chunk.iterrows():
            eng, frc = row["en"], row["fr"]

            if pd.isna(eng) or pd.isna(frc):
                continue 
            
            eng_ids = [sp.bos_id()] + sp.encode(str(eng), out_type=int) + [sp.eos_id()]
            frc_ids = [sp.bos_id()] + sp.encode(str(frc), out_type=int) + [sp.eos_id()]
            eng_data.append(torch.tensor(eng_ids, dtype=torch.long))
            frc_data.append(torch.tensor(frc_ids, dtype=torch.long))

            total +=1 
            
            if total%100000 == 0:
                print(f"At {total} samples")
    
    torch.save({"eng":eng_data, "frc":frc_data}, output_path)

    print(f"Tokenized {total} sentences.")


def find_all_words_in_sentences_with_word(words, invert_index, tokenized, vocab_len):
    return_seq = {}
    for word in words:
        counts = torch.zeros(vocab_len, dtype=torch.int32)
        cont_sent = invert_index["index_en"][word]
        for sent in cont_sent:
            tokens = tokenized["frc"][sent]
            for tok in tokens:
                counts[tok] +=1 

        return_seq[word]=counts
    
    return return_seq



if __name__=="__main__":
    invert_index_dataset_path = "invert_index_path.pt"
    if not os.path.exists(invert_index_dataset_path):
        dataset_path = r"/home/parth/.cache/kagglehub/datasets/dhruvildave/en-fr-translation-dataset/versions/2/en-fr.csv"
        tokenizer_path = r"en_fr.model"
        invert_index(dataset_path, tokenizer_path, invert_index_dataset_path)
    
    invert_index = torch.load(invert_index_dataset_path)

    tokenized_path = "pretokenized.pt"
    if not os.path.exists(tokenized_path):
        dataset_path = r"/home/parth/.cache/kagglehub/datasets/dhruvildave/en-fr-translation-dataset/versions/2/en-fr.csv"
        tokenizer_path = r"en_fr.model"
        pretokenize(dataset_path, tokenizer_path, tokenized_path)
    
    pretokenized = torch.load(tokenized_path)

    vocab_len = 16000


    
    