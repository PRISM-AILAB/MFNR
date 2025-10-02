import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.path import RAW_PATH, PROCESSED_PATH
from src.utils import getDF
from src.bert import BertExtractor


class DataLoader:
    def __init__(self, 
                 fname: str, source: str = "amazon", test_size: float = 0.2,
                 batch_size: int = 8, chunk_size: int = 2048,  
                 use_max_length: bool = False, max_length: int = 512, use_mean_pooling: bool = False,
                 verbose: bool = True):
        
        self.fname = fname
        self.source = source.lower()
        self.test_size = test_size
        
    
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.use_max_length = use_max_length
        self.max_length = max_length
        self.use_mean_pooling = use_mean_pooling
        self.verbose = verbose
        self.bert_extractor = self._load_bert_extractor(model_ckpt = "bert-base-uncased")
        self.robert_extractor = self._load_bert_extractor(model_ckpt = "roberta-base")

        self.raw_df = self._data_loader()
        self.train, self.test = self._data_preprocessor()

    def _data_loader(self):
        if self.source == "amazon":
            fpath = os.path.join(RAW_PATH, f"{self.fname}.jsonl.gz")

            df = getDF(fpath)[["user_id", "asin", "text", "rating"]]
            df = df.rename(columns = {
                "user_id": "user",
                "asin": "item",                
            })
            return df
        
        elif self.source == "yelp":
            raise ValueError("The Yelp source is not implemented yet.") # yelp 추가필요
        
        else:
            raise ValueError("Invalid source: must be either 'amazon' or 'yelp'.")
        
    
    def _label_encoding(self, df: pd.DataFrame, col: str):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].values)
        return df

    def _get_len(self, df: pd.DataFrame, col: str):
        return df[col].nunique()
    
    def _text_aggregator(self, df: pd.DataFrame, col: str):
        agg_df = df.groupby(col)["text"].apply(lambda x: ";".join(x)).reset_index()
        agg_df.columns = [col, f"agg_{col}_text"]
        return agg_df
    
    def _load_bert_extractor(self, model_ckpt):
        return BertExtractor(
            model_ckpt = model_ckpt,
            batch_size = self.batch_size,
            chunk_size = self.chunk_size,
            use_max_length = self.use_max_length,
            use_mean_pooling = self.use_mean_pooling,
            verbose = self.verbose
        )

    def _data_preprocessor(self):
        df = self.raw_df.copy()

        df = df.dropna()
        df = self._label_encoding(df, "user")
        df = self._label_encoding(df, "item")

        print(f"The shape of Total dataset: {df.shape}")
        print(f"The number of users: {self._get_len(df, 'user')}")
        print(f"The number of items: {self._get_len(df, 'item')}")

        user_agg_df = self._text_aggregator(df, "user")
        item_agg_df = self._text_aggregator(df, "item")

        user_agg_df = self.bert_extractor.run(user_agg_df, text_col = "agg_user_text", output_col="user_bert")
        user_agg_df = self.robert_extractor.run(user_agg_df, text_col = "agg_user_text", output_col = "user_roberta")
        item_agg_df = self.bert_extractor.run(item_agg_df, text_col = "agg_item_text", output_col = "item_bert")
        item_agg_df = self.robert_extractor.run(item_agg_df, text_col = "agg_item_text", output_col = "item_roberta")

        joined_df = df.merge(user_agg_df, on = "user", how = "left")
        final_df = joined_df.merge(item_agg_df, on = "item", how = "left")

        train, test = train_test_split(final_df, test_size=self.test_size, random_state=42)
        train_fpath = os.path.join(PROCESSED_PATH, "train.parquet")
        test_fpath = os.path.join(PROCESSED_PATH, "test.parquet")
        train.to_parquet(train_fpath, engine = "pyarrow", index = False)
        test.to_parquet(test_fpath, engine = "pyarrow", index = False)

        return train, test