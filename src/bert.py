import pandas as pd
from tqdm import tqdm
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModel


class BertExtractor:
    def __init__(
        self,
        model_ckpt: str = "bert-base-uncased",
        batch_size: int = 8,
        chunk_size: int = 2048,
        use_max_length: bool = False,
        max_length: int = 512,
        use_mean_pooling: bool = False,
        verbose: bool = True,
    ):
        self.model_ckpt = model_ckpt
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.use_max_length = use_max_length
        self.max_length = max_length
        self.use_mean_pooling = use_mean_pooling
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = self._model()

    def _model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        model = AutoModel.from_pretrained(self.model_ckpt).to(self.device)
        model.eval()
        return tokenizer, model

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1)
        masked = last_hidden * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        return masked.sum(dim=1) / lengths

    @torch.inference_mode()
    def _encode_batch(self, texts: List[str], progress: Optional[object] = None) -> List[List[float]]:
        pad_kwargs = (
            dict(padding="max_length", truncation=True, max_length=self.max_length)
            if self.use_max_length
            else dict(padding=True, truncation=True)  # longest
        )

        embs = []
        rng = range(0, len(texts), self.batch_size)

        for i in rng:
            bt = texts[i : i + self.batch_size]

            inputs = self.tokenizer(bt, return_tensors="pt", **pad_kwargs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [B, T, H]

            if self.use_mean_pooling:
                batch_emb = self._mean_pool(last_hidden, inputs["attention_mask"])
            else:
                batch_emb = last_hidden[:, 0, :]  # CLS

            embs.append(batch_emb.cpu())

            del outputs, last_hidden, batch_emb, inputs

            if progress is not None:
                progress.update(len(bt))

        return torch.cat(embs, dim=0).numpy().tolist()

    def run(self, df: pd.DataFrame, text_col: str, output_col: str = "EMBEDDING") -> pd.DataFrame:
        if text_col not in df.columns:
            raise KeyError(f"{text_col} column not found in DataFrame.")

        df = df.copy()
        df[text_col] = df[text_col].fillna("").astype(str)

        n = len(df)
        parts = []

        pbar = tqdm(total=n, desc="Embedding", unit="rows") if self.verbose else None

        try:
            for start in range(0, n, self.chunk_size):
                end = min(start + self.chunk_size, n)
                temp = df.iloc[start:end].copy()
                texts = temp[text_col].tolist()

                temp_embs = self._encode_batch(texts, progress=pbar)
                assert len(temp_embs) == len(temp), "Embedding length mismatch."

                temp[output_col] = temp_embs
                parts.append(temp)
        finally:
            if pbar is not None:
                pbar.close()

        return pd.concat(parts, axis=0, ignore_index=True)