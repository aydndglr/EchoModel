
# src/dataset/custom_data_loader.py

import json
import logging
import os
from typing import Dict, Any, Union, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset.base_text_dataset import BaseTextDataset
from src.dataset.data_collator import DataCollatorForLanguageModeling
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.char_tokenizer import CharTokenizer # Eğer CharTokenizer da kullanılacaksa

log = logging.getLogger(__name__)

class CustomDataLoader:
    """
    EchoModel için yapılandırılabilir bir veri yükleyici.
    İşlenmiş veri dosyalarını (Parquet/JSONL) yükler, belirtilen tokenizer ile tokenleştirir
    ve PyTorch DataLoader kullanarak batçeler halinde sunar.
    """
    def __init__(self, 
                 data_filepath: Union[str, Path],
                 tokenizer_path: str,
                 max_seq_len: int,
                 batch_size: int,
                 tokenizer_type: str = "bpe", # 'bpe' veya 'char'
                 add_special_tokens: bool = True,
                 text_key: str = "text",
                 num_workers: int = 0, # DataLoader için işçi sayısı
                 shuffle: bool = True,
                 pin_memory: bool = True): # Tensörleri GPU'ya daha hızlı aktarmak için
        """
        Args:
            data_filepath (Union[str, Path]): İşlenmiş veri dosyasının yolu.
            tokenizer_path (str): Tokenizer dosyalarının (örn. tokenizer.json, char_vocab.json) bulunduğu dizin.
            max_seq_len (int): Modelin işleyebileceği maksimum sekans uzunluğu.
            batch_size (int): DataLoader'dan döndürülecek batçe boyutu.
            tokenizer_type (str): Kullanılacak tokenizer tipi ('bpe' veya 'char').
            add_special_tokens (bool): Tokenizasyon sırasında özel tokenların eklenip eklenmeyeceği.
            text_key (str): Veri dosyasındaki metin içeriğini içeren anahtar (örn. "text").
            num_workers (int): DataLoader için kaç işçi sürecinin kullanılacağı.
            shuffle (bool): Her epoch'ta verinin karıştırılıp karıştırılmayacağı.
            pin_memory (bool): Veri yükleyicisinin tensörleri CUDA pinli belleğe kopyalayıp kopyalamayacağı.
                               GPU eğitimi için önerilir.
        """
        self.data_filepath = Path(data_filepath)
        self.tokenizer_path = tokenizer_path
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.tokenizer_type = tokenizer_type
        self.add_special_tokens = add_special_tokens
        self.text_key = text_key
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        self.tokenizer_instance = self._load_tokenizer()
        self.dataset = BaseTextDataset(
            filepath=self.data_filepath,
            tokenizer_instance=self.tokenizer_instance,
            max_seq_len=self.max_seq_len,
            add_special_tokens=self.add_special_tokens,
            text_key=self.text_key
        )
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer_instance=self.tokenizer_instance,
            max_seq_len=self.max_seq_len # Collator'a da max_seq_len veriyoruz
        )
        
        self.dataloader = self._create_dataloader()
        
        log.info(f"CustomDataLoader başlatıldı. Veri seti boyutu: {len(self.dataset)}, Batçe boyutu: {self.batch_size}")

    def _load_tokenizer(self):
        """
        Belirtilen tipte ve yoldan tokenizer'ı yükler.
        """
        if self.tokenizer_type == "bpe":
            return BPETokenizer.from_pretrained(self.tokenizer_path)
        elif self.tokenizer_type == "char":
            # CharTokenizer için bir vocabulary (chars listesi) sağlamamız gerekebilir.
            # Veya 'from_pretrained' metoduyla yükleme yapmalıyız.
            # Şu anki CharTokenizer.from_pretrained, doğrudan `char_vocab.json` bekliyor.
            return CharTokenizer.from_pretrained(self.tokenizer_path)
        else:
            raise ValueError(f"Desteklenmeyen tokenizer tipi: {self.tokenizer_type}. 'bpe' veya 'char' olmalı.")

    def _create_dataloader(self) -> DataLoader:
        """
        PyTorch DataLoader objesini oluşturur.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.data_collator, # Kendi collator'ımızı kullanıyoruz
            pin_memory=self.pin_memory
        )

    def get_dataloader(self) -> DataLoader:
        """
        Hazırlanmış PyTorch DataLoader objesini döndürür.
        """
        return self.dataloader

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("CustomDataLoader testi başlatılıyor...")

    import shutil
    from src.utils.logger import setup_logging
    from src.config.base_config import BaseConfig # max_seq_len ve vocab_size için
    
    test_output_dir = "test_runs_custom_dataloader"
    test_run_name = "custom_dataloader_test_run"
    
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True) 

    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logging.root.removeHandler(handler)
    setup_logging(log_level="INFO", output_dir=test_output_dir, run_name=test_run_name)
    log = logging.getLogger(__name__)

    base_cfg = BaseConfig() # Varsayılan değerlerle
    
    # Adım 1: Dummy tokenizer eğitimi (bpe_tokenizer'ı kullanacağız)
    tokenizer_save_path = Path(test_output_dir) / "bpe_tokenizer_assets"
    os.makedirs(tokenizer_save_path, exist_ok=True)
    temp_tokenizer_train_file = Path(test_output_dir) / "dataloader_tokenizer_train_data.txt"
    with open(temp_tokenizer_train_file, "w", encoding="utf-8") as f:
        f.write("Bu, dataloader testi için kullanılan bir metindir. Tokenizer eğitimi için kullanılır.\n")
        f.write("Daha fazla veri, daha iyi tokenizer. Modelinizi eğitmek için hazır olun.\n")
    
    bpe_tokenizer_instance_for_dl = BPETokenizer(vocab_size=base_cfg.vocab_size)
    bpe_tokenizer_instance_for_dl.train(files=[str(temp_tokenizer_train_file)], save_path=str(tokenizer_save_path))
    
    # Tokenizer'ın pad_token_id'si güncellenmeli (train metodu hallediyor)
    # Eğer varsayılan bir ID gerekiyorsa (eğitimde oluşmadıysa), burada atamalıyız.
    # bpe_tokenizer.py'deki _update_special_token_ids() bunu hallediyor olmalı.
    
    # Adım 2: Dummy işlenmiş veri dosyası (JSONL)
    dummy_data_path = Path(test_output_dir) / "processed_texts_for_dataloader.jsonl"
    with open(dummy_data_path, "w", encoding="utf-8") as f:
        json.dump({"text": "Kısa cümle 1."}, f)
        f.write("\n")
        json.dump({"text": "Uzun bir cümle örneği olarak bu metni kullanacağız."}, f)
        f.write("\n")
        json.dump({"text": "Çok daha kısa bir cümle."}, f)
        f.write("\n")
        json.dump({"text": "Bu, bir başka uzun deneme metnidir. Padding testini de yapalım."}, f)
        f.write("\n")

    # CustomDataLoader oluşturma
    print("\n--- CustomDataLoader Oluşturma ve Yükleme Testi ---")
    data_loader_config = {
        "data_filepath": str(dummy_data_path),
        "tokenizer_path": str(tokenizer_save_path),
        "max_seq_len": 30, # Küçük bir max_seq_len ile padding'i daha net görelim
        "batch_size": 2,
        "tokenizer_type": "bpe",
        "num_workers": 0, # Test için 0 worker daha güvenli
        "shuffle": True,
        "pin_memory": False # Test için False
    }

    try:
        custom_dl = CustomDataLoader(**data_loader_config)
        dataloader = custom_dl.get_dataloader()
        
        print(f"  DataLoader başarıyla oluşturuldu. Örnek sayısı: {len(custom_dl.dataset)}")
        print(f"  Batch boyutu: {dataloader.batch_size}")

        # Batçeleri test et
        print("\n--- Batçe Iterasyon Testi ---")
        first_batch = next(iter(dataloader))
        
        print(f"  İlk batçe input_ids boyutu: {first_batch['input_ids'].shape}")
        print(f"  İlk batçe attention_mask boyutu: {first_batch['attention_mask'].shape}")
        print(f"  İlk batçe labels boyutu: {first_batch['labels'].shape}")

        expected_batch_size = data_loader_config["batch_size"]
        expected_seq_len = data_loader_config["max_seq_len"]

        assert first_batch["input_ids"].shape == (expected_batch_size, expected_seq_len), "input_ids boyutu yanlış!"
        assert first_batch["attention_mask"].shape == (expected_batch_size, expected_seq_len), "attention_mask boyutu yanlış!"
        assert first_batch["labels"].shape == (expected_batch_size, expected_seq_len), "labels boyutu yanlış!"

        # Padding ve attention_mask'i kontrol et
        # Padding token ID'si `pad_token_id` olmalı
        # Labels'da padding `-100` olmalı
        # `attention_mask` padding yerlerinde 0 olmalı

        # Örnek olarak ilk batçedeki bir örneğin paddingini kontrol edebiliriz
        # Bu biraz karmaşık olabilir çünkü shuffle açık ve hangi örneğin hangi sırada geleceğini bilemeyiz.
        # Ancak, batçedeki herhangi bir padding'in doğru değerde olup olmadığını kontrol edebiliriz.
        pad_id = custom_dl.tokenizer_instance.pad_token_id
        
        # input_ids'da padding bölgelerinde pad_id var mı?
        assert (first_batch["input_ids"] == pad_id).any(), "Input_ids'de padding bulunamadı veya yanlış!"
        # labels'da padding bölgelerinde -100 var mı?
        assert (first_batch["labels"] == -100).any(), "Labels'da padding (-100) bulunamadı veya yanlış!"
        # attention_mask'ta padding bölgelerinde 0 var mı?
        assert (first_batch["attention_mask"] == 0).any(), "Attention_mask'ta padding (0) bulunamadı veya yanlış!"


        print("  Batçe iterasyon ve padding testi başarılı. ✅")

    except Exception as e:
        print(f"CustomDataLoader testi BAŞARISIZ: {e} ❌")
        log.error(f"CustomDataLoader testi BAŞARISIZ: {e}", exc_info=True)
    finally:
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)
            log.info(f"Temizlik yapıldı: '{test_output_dir}' silindi.")

    print("\nCustomDataLoader tüm testleri tamamlandı. ✅")