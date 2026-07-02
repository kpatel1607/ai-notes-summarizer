from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class OCRCorrectionModel:
    def __init__(
        self,
        model_name: str = "yelpfeast/byt5-base-english-ocr-correction",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
        ).to(self.device)

        self.model.eval()

    def correct(
        self,
        text: str,
    ) -> Dict[str, Any]:

        cleaned_input = text.strip()

        if not cleaned_input:
            return {
                "corrected_text": "",
                "model_used": "none",
                "confidence": 0.0,
            }

        inputs = self.tokenizer(
            cleaned_input,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
            )

        corrected = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return {
            "corrected_text": corrected.strip(),
            "model_used": "byt5_ocr_correction",
            "confidence": 0.75,
        }