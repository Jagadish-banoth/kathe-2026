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


if __name__ == "__main__":
    model, tokenizer, device = load_model()

    print("Model loaded successfully.")
    print("Base model:", BASE_MODEL)
    print("Adapter:", ADAPTER)
    print("Device:", device)
