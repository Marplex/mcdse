from typing import List, Tuple
import PIL
from datasets import load_dataset
from torch.utils.data import Dataset
from arguments import DataArguments
from utils import smart_resize, smart_resize_2
from PIL import Image
import io


class TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments):
        self.data_args = data_args

        self.factor = 28
        self.min_pixels = 1 * self.factor * self.factor
        self.max_pixels = 960 * self.factor * self.factor

        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )

        self.text_only_training = self.data_args.corpus_path is None

        if not self.text_only_training:
            self.corpus = load_dataset(
                self.data_args.corpus_name,
                self.data_args.corpus_config,
                data_files=self.data_args.corpus_path,
                split=self.data_args.corpus_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )

            self.docid2idx = {docid: i for i, docid in enumerate(self.corpus['docid'])}

    def __len__(self):
        return len(self.train_data)

    def _get_image(self, docid):
        image = self.corpus[self.docid2idx[docid]]['image']['bytes']
        img = Image.open(io.BytesIO(image))
        new_size = smart_resize_2(
            img.height,
            img.width,
            self.factor,
            self.min_pixels,
            self.max_pixels
        )

        return img.convert("RGB").resize(new_size)

    def __getitem__(self, item) -> Tuple[str, List[PIL.Image.Image | str]]:
        group = self.train_data[item]
        formated_query = group['q']
        formated_passages = [group['pos'] if self.text_only_training else self._get_image(group['pos'])]

        return formated_query, formated_passages
