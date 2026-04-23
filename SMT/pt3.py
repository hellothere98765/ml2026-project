import torch
import sentencepiece as spm
import pandas as pd

def build_refinement_csv(
    pt_path,
    csv_path,
    spm_path,
    output_path,
    en_col="en",
    frc_col="fr",
    chunksize=100000
):
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)

    data = torch.load(pt_path)
    top_indices = data["top_indices"]  # (vocab_size, 3)
    count=0
    first = True
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        src_sentences = []
        tgt_sentences = []

        for _, row in chunk.iterrows():
            en_text = str(row[en_col])
            frc_text = str(row[frc_col])

            if pd.isna(row[en_col]) or pd.isna(row[frc_col]):
                continue

            # Tokenize English sentence into token IDs
            en_ids = sp.encode(en_text, out_type=int)

            # For each token, look up its top-3 French neighbors
            gloss_ids = []
            for tok_id in en_ids:
                neighbors = top_indices[tok_id].tolist()  # 3 French token IDs
                gloss_ids.extend(neighbors)

            # Decode gloss IDs back to text
            gloss_text = sp.decode(gloss_ids)

            src_sentences.append(gloss_text)
            tgt_sentences.append(frc_text)

        out_chunk = pd.DataFrame({"fr_gloss": src_sentences, "frc": tgt_sentences})
        out_chunk.to_csv(output_path, mode="a", index=False, header=first)
        first = False
        count+=1
        print(f"Chunk {count} done")

    print(f"Saved to {output_path}")


build_refinement_csv(
    pt_path="high_weights.pt",
    csv_path=r"/home/parth/.cache/kagglehub/datasets/dhruvildave/en-fr-translation-dataset/versions/2/en-fr.csv",
    spm_path="en_fr.model",
    output_path="refinement_data.csv",
)