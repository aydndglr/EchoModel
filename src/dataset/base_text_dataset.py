
# src/dataset/base_text_dataset.py

import os
import torch
from torch.utils.data import Dataset
import logging
import pandas as pd
import json
from typing import List, Dict, Any, Union, Optional
from pathlib import Path

# Tokenizer'ı içeri aktaracağız (bpe_tokenizer veya char_tokenizer)
# Ancak burada somut bir tokenizer kullanmayacağız, sadece arayüz olarak bir "tokenizer" objesi bekleyeceğiz.
# Bu yüzden doğrudan import etmiyoruz, tipi belirterek bekliyoruz.
# from src.tokenizer.bpe_tokenizer import BPETokenizer
# from src.tokenizer.char_tokenizer import CharTokenizer

log = logging.getLogger(__name__)

class BaseTextDataset(Dataset):
    """
    İşlenmiş metin veri setlerini yükleyen, tokenize eden ve
    model girdisi için uygun formatta tensörler döndüren temel PyTorch Dataset sınıfı.
    """
    def __init__(self, 
                 filepath: Union[str, Path], 
                 tokenizer_instance: Any, # BPETokenizer veya CharTokenizer objesi
                 max_seq_len: int,
                 add_special_tokens: bool = True,
                 text_key: str = "text"):
        """
        Args:
            filepath (Union[str, Path]): İşlenmiş veri dosyasının yolu (örn. processed_data.parquet/jsonl).
            tokenizer_instance (Any): Kullanılacak tokenizer objesi (BPETokenizer veya CharTokenizer örneği).
            max_seq_len (int): Modelin işleyebileceği maksimum sekans uzunluğu.
            add_special_tokens (bool): Tokenizasyon sırasında özel tokenların eklenip eklenmeyeceği.
            text_key (str): Veri dosyasındaki metin içeriğini içeren anahtar (örn. "text").
        """
        self.filepath = Path(filepath)
        self.tokenizer = tokenizer_instance
        self.max_seq_len = max_seq_len
        self.add_special_tokens = add_special_tokens
        self.text_key = text_key
        
        self.data: List[Dict[str, Any]] = []
        self._load_data()

        log.info(f"BaseTextDataset başlatıldı. Dosya: '{self.filepath}', Toplam örnek: {len(self.data)}")
        log.info(f"Max sekans uzunluğu: {self.max_seq_len}, Özel tokenlar eklenecek mi: {self.add_special_tokens}")

    def _load_data(self):
        """
        Veri dosyasını yükler (Parquet veya JSONL).
        """
        if not self.filepath.is_file():
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {self.filepath}")

        if self.filepath.suffix == '.parquet':
            try:
                self.data = pd.read_parquet(self.filepath).to_dict(orient='records')
            except ImportError:
                log.error("Parquet okumak için pandas ve pyarrow kütüphaneleri gerekli.")
                raise
        elif self.filepath.suffix == '.jsonl':
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        else:
            raise ValueError(f"Desteklenmeyen veri dosyası formatı: {self.filepath.suffix}. Parquet veya JSONL olmalı.")

    def __len__(self) -> int:
        """Veri setindeki toplam örnek sayısını döndürür."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Veri setinden belirli bir indeksteki örneği döndürür.
        Metni tokenize eder, kırpar ve gerekli tensörlere dönüştürür.
        """
        item = self.data[idx]
        text = item.get(self.text_key, "")
        
        if not text:
            log.warning(f"Örnek {idx} için metin içeriği boş. Atlanıyor.")
            # Boş örnekler için uygun bir strateji izle: atla, veya hata fırlat
            # Şimdilik, boş bir tensör döndürüp collator'da yoksayılmasını bekleyelim
            # veya veri yükleyicisi tarafında filtrelemeyi düşünmeliyiz.
            return {"input_ids": torch.tensor([]), "labels": torch.tensor([])}

        # Metni tokenize et
        # Tokenizasyon sırasında özel tokenlar eklenmeli mi? (BOS/EOS)
        # Modelin kendisi BOS/EOS tokenlarını input_ids'e ekleyecekse burada eklemeyebiliriz.
        # Genellikle, LLM'lerde modelin kendisi bir sonraki kelimeyi tahmin ederken
        # bu tokenlar otomatik olarak eklenir veya modelin eğitimi bu tokenlara göre yapılır.
        # HF modelleri genellikle add_special_tokens=True ile tokenleme yapar.
        tokenized_ids = self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens)
        
        # Diziyi maksimum sekans uzunluğuna göre kırp
        if len(tokenized_ids) > self.max_seq_len:
            tokenized_ids = tokenized_ids[:self.max_seq_len]

        # PyTorch tensörlerine dönüştür
        input_ids = torch.tensor(tokenized_ids, dtype=torch.long)
        
        # Dil modelleme görevi için etiketler (labels)
        # Etiketler, bir sonraki tokeni tahmin etmek için input_ids'in bir kaydırılmış versiyonudur.
        # Örneğin, input_ids = [T1, T2, T3] ise, labels = [T2, T3, <EOS_TOKEN>] veya [T2, T3, <PAD_TOKEN>] olur.
        # Ancak, çoğu LLM eğitiminde, modelden beklenen çıktı `input_ids[1:]` olduğunda,
        # input_ids'in kendisi genellikle hedef olarak kullanılır ve son token dışarıda bırakılır.
        # Veya `input_ids[1:]` etiket, `input_ids[:-1]` girdi olur.
        # Basit bir dil modelleme görevi için, input_ids'in kendisi etiket olarak kullanılabilir
        # ve kayıp hesaplaması sırasında son token dışarıda bırakılır.
        # Veya, maskeli dil modellemesi için etiketler, input_ids'in bir kopyasıdır
        # ve padding tokenları -100 olarak işaretlenir.
        
        labels = input_ids.clone() # Girdinin kendisi etiket olarak (nedeni: sonraki token tahmini)
        
        # Padding için Collator kullanılacak, o yüzden burada padding yapmıyoruz.
        # Collator ayrıca padding token'larının etiketlerini -100 yapacak.

        return {"input_ids": input_ids, "labels": labels}


# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("BaseTextDataset testi başlatılıyor...")

    import shutil
    from src.utils.logger import setup_logging
    # Test için dummy tokenizer ve config'ler
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    from src.config.base_config import BaseConfig
    
    test_output_dir = "test_runs_dataset"
    test_run_name = "base_text_dataset_test_run"
    
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True) 

    # Geçici olarak tüm log handler'larını kapat (Windows PermissionError'ı için)
    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logging.root.removeHandler(handler)
    setup_logging(log_level="INFO", output_dir=test_output_dir, run_name=test_run_name)
    log = logging.getLogger(__name__)

    # Dummy processed data file (JSONL)
    dummy_data_path = Path(test_output_dir) / "processed_texts.jsonl"
    with open(dummy_data_path, "w", encoding="utf-8") as f:
        json.dump({"text": "Bu birinci test cümlesidir. Çok uzun bir metin."}, f)
        f.write("\n")
        json.dump({"text": "İkinci kısa cümle."}, f)
        f.write("\n")
        json.dump({"text": "Üçüncü, biraz daha uzun bir cümle denemesi yapalım."}, f)
        f.write("\n")
        json.dump({"text": ""}, f) # Boş metin

    # Dummy tokenizer (BPE tokenizer'ı eğitip kullanalım)
    tokenizer_save_path = Path(test_output_dir) / "tokenizer_assets_for_dataset"
    os.makedirs(tokenizer_save_path, exist_ok=True)
    
    # Test için küçük bir metin dosyası oluştur (bpe_tokenizer.py'nin ihtiyacı)
    temp_tokenizer_train_file = Path(test_output_dir) / "tokenizer_train_data.txt"
    with open(temp_tokenizer_train_file, "w", encoding="utf-8") as f:
        f.write("Bu, tokenizer eğitimi için kullanılan bir metindir. Bir iki üç dört beş.\n")
    
    bpe_tokenizer_instance = BPETokenizer(vocab_size=100)
    bpe_tokenizer_instance.train(files=[str(temp_tokenizer_train_file)], save_path=str(tokenizer_save_path))
    
    # BaseConfig'ten max_seq_len alalım
    base_cfg = BaseConfig()
    max_seq_len_test = base_cfg.max_seq_len # 512

    # Dataset oluşturma
    print("\n--- BaseTextDataset Oluşturma Testi ---")
    dataset = BaseTextDataset(
        filepath=dummy_data_path,
        tokenizer_instance=bpe_tokenizer_instance,
        max_seq_len=max_seq_len_test,
        add_special_tokens=True
    )
    print(f"  Veri setindeki toplam örnek: {len(dataset)}")
    assert len(dataset) == 4, "Veri setindeki örnek sayısı yanlış!"
    print("BaseTextDataset oluşturma testi başarılı. ✅")

    # Getitem testi
    print("\n--- Getitem ve Tokenizasyon Testi ---")
    sample_0 = dataset[0] # İlk örnek
    print(f"  Örnek 0 - input_ids boyutu: {sample_0['input_ids'].shape}")
    print(f"  Örnek 0 - labels boyutu: {sample_0['labels'].shape}")
    assert "input_ids" in sample_0 and "labels" in sample_0, "Örnek eksik anahtarlar içeriyor!"
    assert sample_0["input_ids"].ndim == 1, "input_ids tek boyutlu olmalı!"
    assert sample_0["input_ids"].shape == sample_0["labels"].shape, "input_ids ve labels boyutları eşleşmiyor!"
    assert sample_0["input_ids"].max() < bpe_tokenizer_instance.vocabulary_size, "Token ID'leri vocab_size'ı aşıyor!"
    assert len(sample_0["input_ids"]) <= max_seq_len_test, "Sekans uzunluğu max_seq_len'i aşıyor!"
    
    # Boş metin örneğini test et (idx=3)
    sample_3 = dataset[3]
    assert sample_3["input_ids"].numel() == 0, "Boş metin için input_ids boş olmalı!"
    print("  Boş metin örneği işleme testi başarılı.")

    print("Getitem ve Tokenizasyon testi başarılı. ✅")
    
    # Temizlik
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
        log.info(f"Temizlik yapıldı: '{test_output_dir}' silindi.")

    print("\nBaseTextDataset tüm testleri tamamlandı. ✅")