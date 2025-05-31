
# src/dataset/data_collator.py

import os
from pathlib import Path
import torch
import logging
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union


# special_tokens'dan PAD_TOKEN'ı ve UNK_TOKEN'ı alacağız
from src.tokenizer.special_tokens import PAD_TOKEN, UNK_TOKEN 

log = logging.getLogger(__name__)

class DataCollatorForLanguageModeling:
    """
    Dil modelleme görevi için örnekleri bir batçede birleştirir ve padding uygular.
    """
    def __init__(self, tokenizer_instance: Any, mlm: bool = False, max_seq_len: Optional[int] = None):
        """
        Args:
            tokenizer_instance (Any): Kullanılacak tokenizer objesi (BPETokenizer veya CharTokenizer örneği).
                                    Bu objenin `pad_token_id` ve `unk_token_id` özelliklerine sahip olması beklenir.
            mlm (bool): Masked Language Modeling (MLM) için mi yoksa Causal Language Modeling (CLM) için mi.
                        Şimdilik sadece CLM (yani False) destekleniyor.
            max_seq_len (Optional[int]): Batçe başına maksimum sekans uzunluğu. Eğer None ise,
                                        batçedeki en uzun sekansın uzunluğu kullanılır.
        """
        self.tokenizer = tokenizer_instance
        self.mlm = mlm
        self.max_seq_len = max_seq_len

        # Tokenizer'dan PAD token ID'sini al. Eğer yoksa hata fırlat veya varsayılan ata.
        self.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        if self.pad_token_id is None:
            # Tokenizer'ın pad_token_id'si yoksa, bunu BaseConfig'ten alabiliriz
            # veya kendi tokenizer'ımızda default olarak belirleyebiliriz.
            # Şimdilik, eğer yoksa 0 veya vocab_size-1 gibi bir varsayılan atayalım.
            # Ancak tokenizer'ın bu ID'yi doğru şekilde tanımlaması beklenir.
            # log.warning("Tokenizer'da 'pad_token_id' bulunamadı. Lütfen tokenizer'ın doğru yapılandırıldığından emin olun.")
            # Geçici olarak özel token ID'sini doğrudan çağırabiliriz
            # Bu, tokenizasyon adımında halledilmesi gereken bir bağımlılık.
            self.pad_token_id = self.tokenizer.token_to_id(PAD_TOKEN)
            if self.pad_token_id is None:
                raise ValueError(f"Tokenizer'da PAD_TOKEN ('{PAD_TOKEN}') için ID bulunamadı.")


        if self.mlm:
            raise NotImplementedError("MLM (Masked Language Modeling) henüz desteklenmiyor. Sadece CLM desteklenir.")
        
        log.info(f"DataCollatorForLanguageModeling başlatıldı. Pad Token ID: {self.pad_token_id}")


    def __call__(self, examples: List[Dict[str, Union[torch.Tensor, List[int]]]]) -> Dict[str, torch.Tensor]:
        """
        Girdi örneklerini alır, batçeleri oluşturur ve padding uygular.

        Args:
            examples (List[Dict[str, Union[torch.Tensor, List[int]]]):
                PyTorch Dataset'ten gelen örneklerin listesi.
                Her örnek {"input_ids": tensör/liste, "labels": tensör/liste} içerir.

        Returns:
            Dict[str, torch.Tensor]: Batçe halinde işlenmiş tensörler.
                                    {"input_ids": padded tensör, "attention_mask": padding maskesi, "labels": padded etiketler}
        """
        # Boş örnekleri filtrele
        examples = [ex for ex in examples if ex["input_ids"].numel() > 0]
        if not examples:
            return {
                "input_ids": torch.tensor([]),
                "attention_mask": torch.tensor([]),
                "labels": torch.tensor([])
            } 

        # Batçedeki en uzun sekans uzunluğunu bul
        if self.max_seq_len is None:
            max_length = max(len(e["input_ids"]) for e in examples)
        else:
            max_length = self.max_seq_len
        
        # input_ids ve labels için padding uygula
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for example in examples:
            input_ids = example["input_ids"]
            labels = example["labels"] # BaseTextDataset'ten geldiği gibi

            current_len = len(input_ids)
            padding_len = max_length - current_len

            # input_ids'e padding uygula
            padded_input_ids = F.pad(input_ids, (0, padding_len), value=self.pad_token_id)
            batch_input_ids.append(padded_input_ids)

            # labels'a padding uygula
            # Dil modelleme için padding tokenlarının etiketlerini -100 yapıyoruz.
            # Bu sayede CrossEntropyLoss bu konumları kayıp hesaplamasına dahil etmez.
            padded_labels = F.pad(labels, (0, padding_len), value=-100)
            batch_labels.append(padded_labels)

            # attention_mask oluştur
            # Gerçek tokenler için 1, padding için 0
            attention_mask = torch.cat([
                torch.ones(current_len, dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long)
            ])
            batch_attention_mask.append(attention_mask)
        
        # Tensörleri yığınla
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels)
        }

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("DataCollatorForLanguageModeling testi başlatılıyor...")

    import shutil
    from src.utils.logger import setup_logging
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    from src.tokenizer.special_tokens import PAD_TOKEN # PAD_TOKEN direkt import etmeliyiz
    
    test_output_dir = "test_runs_data_collator"
    test_run_name = "collator_test_run"
    
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True) 

    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logging.root.removeHandler(handler)
    setup_logging(log_level="INFO", output_dir=test_output_dir, run_name=test_run_name)
    log = logging.getLogger(__name__)

    # Dummy tokenizer (pad_token_id olması önemli)
    # BPE tokenizer'ı eğitip kullanalım ki pad_token_id'si olsun.
    tokenizer_save_path = Path(test_output_dir) / "tokenizer_assets_for_collator"
    os.makedirs(tokenizer_save_path, exist_ok=True)
    temp_tokenizer_train_file = Path(test_output_dir) / "collator_tokenizer_train_data.txt"
    with open(temp_tokenizer_train_file, "w", encoding="utf-8") as f:
        f.write("Bu, collator testi için bir metindir. Padding tokenı önemlidir.\n")
    
    bpe_tokenizer_instance = BPETokenizer(vocab_size=100) # Küçük vocab
    bpe_tokenizer_instance.train(files=[str(temp_tokenizer_train_file)], save_path=str(tokenizer_save_path))
    
    # tokenizer_instance'ın pad_token_id'ye sahip olduğundan emin olalım
    pad_token_id_from_tokenizer = bpe_tokenizer_instance.token_to_id(PAD_TOKEN)
    if pad_token_id_from_tokenizer is None:
        # Eğer eğitimde özel tokenlar eklenmediyse, manuel olarak bir ID atayalım
        log.warning(f"PAD_TOKEN ID'si bulunamadı, varsayılan olarak {bpe_tokenizer_instance.vocabulary_size} atanıyor.")
        setattr(bpe_tokenizer_instance, 'pad_token_id', bpe_tokenizer_instance.vocabulary_size)
    else:
        setattr(bpe_tokenizer_instance, 'pad_token_id', pad_token_id_from_tokenizer)

    # Örnek `examples` listesi (BaseTextDataset'ten gelen format)
    # Her bir örnek, `input_ids` (tensör) ve `labels` (tensör) içerir.
    examples_input = [
        {"input_ids": torch.tensor([1, 2, 3, 4, 5]), "labels": torch.tensor([1, 2, 3, 4, 5])},
        {"input_ids": torch.tensor([6, 7, 8]), "labels": torch.tensor([6, 7, 8])},
        {"input_ids": torch.tensor([9, 10, 11, 12, 13, 14, 15, 16]), "labels": torch.tensor([9, 10, 11, 12, 13, 14, 15, 16])},
        {"input_ids": torch.tensor([]), "labels": torch.tensor([])} # Boş örnek
    ]

    # DataCollator oluştur
    collator = DataCollatorForLanguageModeling(tokenizer_instance=bpe_tokenizer_instance, max_seq_len=10) # max_seq_len belirleyelim

    print("\n--- Batçe Oluşturma ve Padding Testi ---")
    batched_data = collator(examples_input)

    print(f"  input_ids boyutu: {batched_data['input_ids'].shape}")
    print(f"  attention_mask boyutu: {batched_data['attention_mask'].shape}")
    print(f"  labels boyutu: {batched_data['labels'].shape}")

    # Beklenen boyutları kontrol et
    expected_batch_size = 3 # Boş örnek filtrelendiği için
    expected_seq_len = 10 # max_seq_len olarak belirlendiği için
    assert batched_data["input_ids"].shape == (expected_batch_size, expected_seq_len), "input_ids batçe boyutu yanlış!"
    assert batched_data["attention_mask"].shape == (expected_batch_size, expected_seq_len), "attention_mask batçe boyutu yanlış!"
    assert batched_data["labels"].shape == (expected_batch_size, expected_seq_len), "labels batçe boyutu yanlış!"

    # Padding değerlerini kontrol et
    # İlk örnek (5 elemanlı) -> 5 gerçek, 5 padding
    # İkinci örnek (3 elemanlı) -> 3 gerçek, 7 padding
    # Üçüncü örnek (8 elemanlı) -> 8 gerçek, 2 padding
    assert torch.all(batched_data["input_ids"][0, 5:] == bpe_tokenizer_instance.pad_token_id), "İlk örnekte input_ids padding yanlış!"
    assert torch.all(batched_data["labels"][0, 5:] == -100), "İlk örnekte labels padding yanlış!"
    assert torch.all(batched_data["attention_mask"][0, :5] == 1) and torch.all(batched_data["attention_mask"][0, 5:] == 0), "İlk örnekte attention_mask yanlış!"

    assert torch.all(batched_data["input_ids"][1, 3:] == bpe_tokenizer_instance.pad_token_id), "İkinci örnekte input_ids padding yanlış!"
    assert torch.all(batched_data["labels"][1, 3:] == -100), "İkinci örnekte labels padding yanlış!"
    assert torch.all(batched_data["attention_mask"][1, :3] == 1) and torch.all(batched_data["attention_mask"][1, 3:] == 0), "İkinci örnekte attention_mask yanlış!"
    
    assert torch.all(batched_data["input_ids"][2, 8:] == bpe_tokenizer_instance.pad_token_id), "Üçüncü örnekte input_ids padding yanlış!"
    assert torch.all(batched_data["labels"][2, 8:] == -100), "Üçüncü örnekte labels padding yanlış!"
    assert torch.all(batched_data["attention_mask"][2, :8] == 1) and torch.all(batched_data["attention_mask"][2, 8:] == 0), "Üçüncü örnekte attention_mask yanlış!"

    print("Batçe oluşturma ve padding testi başarılı. ✅")

    # Temizlik
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
        log.info(f"Temizlik yapıldı: '{test_output_dir}' silindi.")

    print("\nDataCollatorForLanguageModeling tüm testleri tamamlandı. ✅")