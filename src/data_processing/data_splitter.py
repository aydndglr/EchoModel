
# src/data_processing/data_splitter.py

import json
import os
import logging
import random
import pandas as pd
from typing import List, Tuple, Dict, Any, Union
from pathlib import Path

log = logging.getLogger(__name__)

class DataSplitter:
    """
    İşlenmiş veri setlerini eğitim, doğrulama ve test kümelerine ayıran sınıf.
    Parquet veya JSONL gibi dosya formatlarını destekler.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(self.seed)
        log.info(f"DataSplitter başlatıldı. Tohum: {self.seed}")

    def split_dataset(
        self,
        input_filepath: Path,
        output_dir: Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True
    ) -> Dict[str, Path]:
        """
        Girdi dosyasını eğitim, doğrulama ve test setlerine ayırır.
        Ayırılan setleri belirtilen çıktı dizinine kaydeder.

        Args:
            input_filepath (Path): Ayrıştırılacak veri dosyasının yolu (örn. processed_data.parquet/jsonl).
            output_dir (Path): Ayrılan veri setlerinin kaydedileceği dizin.
            train_ratio (float): Eğitim setinin oranı.
            val_ratio (float): Doğrulama setinin oranı.
            test_ratio (float): Test setinin oranı.
            shuffle (bool): Veriyi bölmeden önce karıştırıp karıştırmayacağı.

        Returns:
            Dict[str, Path]: Her bir split'in kaydedildiği dosya yollarını içeren sözlük.
                             Örn: {"train": Path_to_train, "val": Path_to_val, "test": Path_to_test}
        """
        if not input_filepath.is_file():
            raise FileNotFoundError(f"Girdi dosyası bulunamadı: {input_filepath}")
        
        if not (train_ratio + val_ratio + test_ratio == 1.0):
            raise ValueError("Eğitim, doğrulama ve test oranlarının toplamı 1.0 olmalıdır.")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"'{input_filepath}' dosyasını bölmeye başlanıyor...")

        # Veriyi yükle
        data: List[Dict[str, Any]] = []
        if input_filepath.suffix == '.parquet':
            try:
                # Pandas ve PyArrow yüklü olmalı
                data = pd.read_parquet(input_filepath).to_dict(orient='records')
            except ImportError:
                log.error("Parquet okumak için pandas ve pyarrow kütüphaneleri gerekli.")
                raise
        elif input_filepath.suffix == '.jsonl':
            with open(input_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {input_filepath.suffix}. Parquet veya JSONL olmalı.")

        if shuffle:
            random.shuffle(data)
            log.info("Veri karıştırıldı.")

        total_samples = len(data)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        split_filepaths = {}
        base_filename = input_filepath.stem # Uzantısız dosya adı

        # Eğitim setini kaydet
        train_output_path = output_dir / f"{base_filename}_train.parquet"
        self._save_split(train_data, train_output_path)
        split_filepaths["train"] = train_output_path

        # Doğrulama setini kaydet
        val_output_path = output_dir / f"{base_filename}_val.parquet"
        self._save_split(val_data, val_output_path)
        split_filepaths["val"] = val_output_path
        
        # Test setini kaydet (eğer oranı > 0 ise)
        if test_ratio > 0 and len(test_data) > 0:
            test_output_path = output_dir / f"{base_filename}_test.parquet"
            self._save_split(test_data, test_output_path)
            split_filepaths["test"] = test_output_path
        
        log.info(f"Veri seti başarıyla bölündü. "
                 f"Eğitim: {len(train_data)} örnek, "
                 f"Doğrulama: {len(val_data)} örnek, "
                 f"Test: {len(test_data)} örnek.")
        
        return split_filepaths

    def _save_split(self, data: List[Dict[str, Any]], filepath: Path):
        """
        Bir veri bölümünü Parquet veya JSONL dosyasına kaydeder.
        """
        if filepath.suffix == '.parquet':
            pd.DataFrame(data).to_parquet(filepath, index=False)
            log.info(f"Veri bölümü kaydedildi: {filepath}")
        elif filepath.suffix == '.jsonl':
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            log.info(f"Veri bölümü kaydedildi: {filepath}")
        else:
            raise ValueError(f"Kaydetmek için desteklenmeyen dosya formatı: {filepath.suffix}")

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("DataSplitter testi başlatılıyor...")

    import shutil
    from src.utils.logger import setup_logging
    
    test_output_dir = "test_runs_data_splitter"
    test_run_name = "data_splitter_test_run"
    
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True) 

    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logging.root.removeHandler(handler)
    setup_logging(log_level="INFO", output_dir=test_output_dir, run_name=test_run_name)
    log = logging.getLogger(__name__)

    # Test için dummy işlenmiş veri dosyası oluştur (Parquet)
    dummy_processed_data_path = Path(test_output_dir) / "all_processed_data.parquet"
    dummy_data_records = [{"text": f"Bu {i}. örnektir."} for i in range(100)] # 100 örnek
    try:
        pd.DataFrame(dummy_data_records).to_parquet(dummy_processed_data_path, index=False)
        print(f"Dummy Parquet dosyası oluşturuldu: {dummy_processed_data_path}")
    except ImportError:
        print("Pandas veya PyArrow yüklü değil, dummy Parquet oluşturulamadı. JSONL olarak devam edilecek.")
        dummy_processed_data_path = Path(test_output_dir) / "all_processed_data.jsonl"
        with open(dummy_processed_data_path, 'w', encoding='utf-8') as f:
            for record in dummy_data_records:
                f.write(json.dumps(record) + '\n')

    splitter = DataSplitter(seed=123)
    
    print("\n--- Veri Seti Bölme Testi ---")
    try:
        split_paths = splitter.split_dataset(
            input_filepath=dummy_processed_data_path,
            output_dir=Path(test_output_dir) / "splits",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            shuffle=True
        )
        
        print(f"Oluşturulan split yolları: {split_paths}")
        
        assert "train" in split_paths and split_paths["train"].is_file(), "Eğitim seti oluşturulamadı!"
        assert "val" in split_paths and split_paths["val"].is_file(), "Doğrulama seti oluşturulamadı!"
        assert "test" in split_paths and split_paths["test"].is_file(), "Test seti oluşturulamadı!"

        # Boyutları kontrol et
        train_len = len(pd.read_parquet(split_paths["train"])) if split_paths["train"].suffix == ".parquet" else len(list(open(split_paths["train"])))
        val_len = len(pd.read_parquet(split_paths["val"])) if split_paths["val"].suffix == ".parquet" else len(list(open(split_paths["val"])))
        test_len = len(pd.read_parquet(split_paths["test"])) if split_paths["test"].suffix == ".parquet" else len(list(open(split_paths["test"])))
        
        total_len = train_len + val_len + test_len
        print(f"Toplam örnek: {total_len}. Train: {train_len}, Val: {val_len}, Test: {test_len}")
        assert total_len == len(dummy_data_records), "Toplam örnek sayısı yanlış!"
        assert train_len == 70, f"Eğitim seti boyutu yanlış: {train_len}"
        assert val_len == 15, f"Doğrulama seti boyutu yanlış: {val_len}"
        assert test_len == 15, f"Test seti boyutu yanlış: {test_len}"

        print("Veri seti bölme testi başarılı. ✅")

    except Exception as e:
        print(f"Veri seti bölme testi BAŞARISIZ: {e} ❌")
        log.error(f"Veri seti bölme testi BAŞARISIZ: {e}", exc_info=True)
    finally:
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)
            log.info(f"Temizlik yapıldı: '{test_output_dir}' silindi.")

    print("\nDataSplitter tüm testleri tamamlandı. ✅")