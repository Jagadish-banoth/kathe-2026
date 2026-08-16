import argparse

import torch
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


def translate(
    text,
    model,
    tokenizer,
    device,
    num_beams=4,
    max_length=128
):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                "kas_Arab"
            ),
            max_length=max_length,
            num_beams=num_beams
        )

    translation = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return translation.strip()


def main():

    parser = argparse.ArgumentParser(
        description="English → Kashmiri translation"
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="English sentence to translate"
    )

    args = parser.parse_args()

    print("Loading model...")

    model, tokenizer, device = load_model()

    translation = translate(
        args.text,
        model,
        tokenizer,
        device
    )

    print("\nEnglish:")
    print(args.text)

    print("\nKashmiri:")
    print(translation)


if __name__ == "__main__":
    main()
