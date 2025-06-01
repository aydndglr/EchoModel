# src/tokenizer/bpe_tokenizer.py

import os
import json
import logging
import shutil
from typing import Dict, List, Optional, Union
from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders
from tokenizers.trainers import BpeTrainer

# src/tokenizer/special_tokens.py dosyasından özel tokenları içeri aktar
from src.tokenizer.special_tokens import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN

log = logging.getLogger(__name__)

class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) tabanlı bir tokenleyiciyi yönetir.
    Tokenleyiciyi eğitebilir, kaydedebilir, yükleyebilir ve metinleri tokenize edip detokenize edebilir.
    """
    def __init__(self, vocab_size: int, unk_token: str = UNK_TOKEN):
        """
        Args:
            vocab_size (int): Tokenleyici kelime haznesinin maksimum boyutu.
            unk_token (str): Bilinmeyen token için kullanılacak string.
        """
        self.tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.special_tokens = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]

        # Temel ön-tokenleyici (whitespace ve punctuation'a göre ayırır)
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        # Post-processor (özel tokenları otomatik eklemek için)
        # Bu kısım, Hugging Face transformer pipeline'ları ile uyumluluk için önemlidir.
        # Örneğin, [BOS_TOKEN] metin [EOS_TOKEN] gibi bir yapı oluşturabilir.
        # Şimdilik manuel ekleme için add_special_tokens=False ile kullanacağız.
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        # Decoder (ID'lerden metne dönüştürmek için)
        self.tokenizer.decoder = decoders.ByteLevel()

        log.info(f"BPETokenizer başlatıldı. Hedef kelime haznesi boyutu: {vocab_size}")

    def train(self, files: Union[str, List[str]], save_path: str):
        """
        Verilen dosyalardan yeni bir BPE tokenleyici eğitir.

        Args:
            files (Union[str, List[str]]): Eğitim için kullanılacak metin dosyalarının yolu veya listesi.
            save_path (str): Eğitilen tokenleyici dosyalarının (tokenizer.json) kaydedileceği dizin.
        """
        if isinstance(files, str):
            files = [files]
        
        log.info(f"Tokenleyici eğitimi başlatılıyor. Dosyalar: {files}")
        log.info(f"Özel tokenlar: {self.special_tokens}")

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2, # Bir tokenın kelime haznesine girmesi için minimum frekans
            special_tokens=self.special_tokens,
            limit_alphabet=1000, # BPE'nin başlaması için başlangıç alfabesi limiti
            show_progress=True
        )
        self.tokenizer.train(files, trainer)
        
        # Eğitilen tokenizer'ı kaydet
        tokenizer_file = os.path.join(save_path, "tokenizer.json")
        self.tokenizer.save(tokenizer_file)
        log.info(f"Tokenleyici başarıyla eğitildi ve '{tokenizer_file}' konumuna kaydedildi.")

        # Eğitildikten sonra özel token ID'lerini güncelleyebiliriz
        self._update_special_token_ids()


    @classmethod
    def from_pretrained(cls, path: str) -> "BPETokenizer":
        """
        Önceden eğitilmiş bir BPE tokenleyiciyi yükler.

        Args:
            path (str): Tokenleyici dosyasının (tokenizer.json) bulunduğu dizin veya dosya yolu.

        Returns:
            BPETokenizer: Yüklenen tokenleyici objesi.
        """
        tokenizer_file = os.path.join(path, "tokenizer.json") if os.path.isdir(path) else path
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"Tokenleyici dosyası bulunamadı: {tokenizer_file}")
        
        tokenizer = Tokenizer.from_file(tokenizer_file)
        
        # Yüklenen tokenizer'ın config'lerini otomatik olarak doldurmak için:
        current_unk_token = tokenizer.model.unk_token if hasattr(tokenizer.model, 'unk_token') else UNK_TOKEN
        current_vocab_size = tokenizer.get_vocab_size()

        instance = cls(vocab_size=current_vocab_size, unk_token=current_unk_token)
        instance.tokenizer = tokenizer # Yüklenen tokenizer objesini ata
        log.info(f"Tokenleyici başarıyla yüklendi: {tokenizer_file}. Kelime haznesi boyutu: {current_vocab_size}")
        
        # Yükleme sonrası özel token ID'lerini güncelle
        instance._update_special_token_ids()

        return instance

    def _update_special_token_ids(self):
        """
        Tokenleyici eğitildikten veya yüklendikten sonra özel tokenların ID'lerini günceller.
        """
        self.bos_token_id = self.tokenizer.token_to_id(BOS_TOKEN)
        self.eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN)
        self.pad_token_id = self.tokenizer.token_to_id(PAD_TOKEN)
        self.unk_token_id = self.tokenizer.token_to_id(UNK_TOKEN)
        log.info(f"Special Token ID'leri güncellendi: BOS={self.bos_token_id}, EOS={self.eos_token_id}, PAD={self.pad_token_id}, UNK={self.unk_token_id}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Metni token ID'lerine dönüştürür.
        """
        encoded = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoded.ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Token ID'lerini metne dönüştürür.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def token_to_id(self, token: str) -> Optional[int]:
        """
        Bir token string'ini ID'sine dönüştürür.
        """
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> Optional[str]:
        """
        Bir token ID'sini string'ine dönüştürür.
        """
        return self.tokenizer.id_to_token(id)

    @property
    def vocabulary_size(self) -> int:
        """Kelime haznesi boyutunu döndürür."""
        return self.tokenizer.get_vocab_size()

    @property
    def get_special_token_ids(self) -> Dict[str, int]:
        """Özel tokenların isimlerini ve ID'lerini döndürür."""
        return {
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
        }

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("BPETokenizer testi başlatılıyor...")

    # Loglama sistemini kur (test için gerekli)
    from src.utils.logger import setup_logging
    test_output_dir = "test_runs_bpe_tokenizer" # Klasör adını daha spesifik yaptık
    test_run_name = "bpe_tokenizer_test_run"
    
    # Mevcut test dizinlerini temizle
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

    # Test için veri dosyası oluştur (Gerçek metinlerle)
    test_data_path = os.path.join(test_output_dir, "test_data.txt")
    with open(test_data_path, "w", encoding="utf-8") as f:
        f.write("Merhaba dünya. Bu bir BPE tokenizer testidir.\n")
        f.write("Transformers kütüphanesi çok kullanışlıdır.\n")
        f.write("Allen Institute for AI (AI2) OLMO'yu geliştirdi.\n")
        f.write("PyTorch ile derin öğrenme projeleri geliştirmek harika.\n")

    # Tokenleyici eğitme
    print("\n--- Tokenleyici Eğitme Testi ---")
    tokenizer_save_path = os.path.join(test_output_dir, "tokenizer_assets")
    os.makedirs(tokenizer_save_path, exist_ok=True)
    
    bpe_tokenizer = BPETokenizer(vocab_size=100) # Daha küçük bir vocab_size ile hızlı test
    bpe_tokenizer.train(files=[test_data_path], save_path=tokenizer_save_path)
    
    # Eğitilen tokenizer'ın kaydedildiğini kontrol et
    assert os.path.exists(os.path.join(tokenizer_save_path, "tokenizer.json")), "Tokenleyici kaydedilemedi!"
    print("Tokenleyici eğitim testi başarılı. ✅")

    # Tokenleyici yükleme ve kullanma
    print("\n--- Tokenleyici Yükleme ve Kullanım Testi ---")
    try:
        loaded_tokenizer = BPETokenizer.from_pretrained(tokenizer_save_path)
        assert loaded_tokenizer.vocabulary_size > 0, "Yüklenen tokenleyici boş!"
        assert loaded_tokenizer.unk_token == UNK_TOKEN, "UNK token yanlış!"
        
        test_text = "Merhaba dünya! Bu yeni bir test cümlesi."
        encoded_ids = loaded_tokenizer.encode(test_text, add_special_tokens=False)
        decoded_text = loaded_tokenizer.decode(encoded_ids, skip_special_tokens=False)
        
        print(f"  Orijinal Metin: '{test_text}'")
        print(f"  Kodlanmış ID'ler: {encoded_ids}")
        print(f"  Çözülmüş Metin: '{decoded_text}'")

        # Özel token ID'lerini test et
        special_ids = loaded_tokenizer.get_special_token_ids
        print(f"  Özel Token ID'leri: {special_ids}")
        assert special_ids["bos_token_id"] is not None, "BOS token ID boş!"
        assert special_ids["eos_token_id"] is not None, "EOS token ID boş!"
        assert special_ids["pad_token_id"] is not None, "PAD token ID boş!"
        assert special_ids["unk_token_id"] is not None, "UNK token ID boş!"
        
        # Test: encode ve decode'un tutarlılığı
        # ByteLevel post-processor genellikle baştaki boşlukları kaldırır veya ekler, bu yüzden strip kullanmak iyi.
        assert decoded_text.strip() == test_text.strip() or (" " + decoded_text).strip() == test_text.strip(), "Encode/Decode tutarsızlığı!"
        
        # Test: özel tokenları ekleme
        text_with_special = f"{BOS_TOKEN} test cümlesi {EOS_TOKEN}"
        encoded_special = loaded_tokenizer.encode(text_with_special, add_special_tokens=False)
        # `add_special_tokens=False` dedik ama manuel eklemiş olsaydık bu id'ler görünmeliydi.
        # Bu test daha çok tokenizasyon mantığını doğrular.
        # Eğer add_special_tokens=True olursa, post_processor'ın davranışına bağlı olur.
        print(f"  Özel tokenlarla (manuel) kodlanmış: {encoded_special}")

        print("Tokenleyici yükleme ve kullanım testi başarılı. ✅")

    except Exception as e:
        print(f"Tokenleyici yükleme veya kullanım testi BAŞARISIZ: {e} ❌")
        # Ensure test directory exists if not already
        os.makedirs(test_output_dir, exist_ok=True)
        # Create a dummy file to ensure directory exists for the finally block
        with open(os.path.join(test_output_dir, "dummy_error_file.txt"), "w") as f:
            f.write("Error occurred during test.")

    finally:
        # Oluşturulan test dizinini temizle (PermissionError'ı önlemek için burada değil, ana test betiğinin sonunda olacak)
        pass # Genel temizlik ana run_all_component_tests() içinde olacak.

    print("\nBPETokenizer tüm testleri tamamlandı. ✅")