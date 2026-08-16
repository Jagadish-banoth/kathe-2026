import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


BASE_MODEL = "facebook/nllb-200-distilled-600M"
ADAPTER = "JagadishBanoth/nllb-en-kas-lora"


def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = (
        torch.float16
        if device == "cuda"
        else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        src_lang="eng_Latn"
    )

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype
    ).to(device)

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER
    )

    model.eval()

    return model, tokenizer, device


def translate_batch(
    texts,
    model,
    tokenizer,
    device,
    num_beams=4,
    max_length=128
):

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                "kas_Arab"
            ),
            max_length=max_length,
            num_beams=num_beams
        )

    return tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )


def main():

    parser = argparse.ArgumentParser(
        description="Batch English → Kashmiri translation"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV containing a sentence column"
    )

    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Output CSV path"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if "sentence" not in df.columns:
        raise ValueError(
            "Input CSV must contain a 'sentence' column."
        )

    print("Loading model...")

    model, tokenizer, device = load_model()

    predictions = []

    for start in tqdm(
        range(0, len(df), args.batch_size),
        desc="English → Kashmiri"
    ):

        texts = (
            df["sentence"]
            .iloc[start:start + args.batch_size]
            .fillna("")
            .astype(str)
            .tolist()
        )

        batch_predictions = translate_batch(
            texts,
            model,
            tokenizer,
            device
        )

        predictions.extend(
            [p.strip() for p in batch_predictions]
        )

    result = df.copy()

    result["kashmiri_text"] = predictions

    result.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTranslation completed.")
    print("Input rows:", len(df))
    print("Output:", args.output)


if __name__ == "__main__":
    main()
