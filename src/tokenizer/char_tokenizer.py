
# src/tokenizer/char_tokenizer.py

import os
import json
import logging
import shutil
from typing import List, Dict, Optional, Set, Union

from src.tokenizer.special_tokens import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN

log = logging.getLogger(__name__)

class CharTokenizer:
    """
    Karakter bazlı bir tokenleyiciyi yönetir.
    Her karakteri benzersiz bir ID'ye eşler. BPE'ye göre daha basit ve küçük kelime haznesi için uygundur.
    """
    def __init__(self, chars: Optional[Union[List[str], Set[str]]] = None, unk_token: str = UNK_TOKEN):
        """
        Args:
            chars (Optional[Union[List[str], Set[str]]]): Tokenleyiciye dahil edilecek karakterlerin listesi veya kümesi.
                                                          Belirtilmezse, boş bir tokenleyici oluşturulur.
            unk_token (str): Bilinmeyen token için kullanılacak string.
        """
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.unk_token = unk_token
        self.special_tokens = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]

        # Özel tokenları her zaman kelime haznesine ekle ve başlangıç ID'lerini ata
        self._add_special_tokens_to_vocab()

        if chars is not None:
            self._build_vocab_from_chars(chars)

        log.info(f"CharTokenizer başlatıldı. Kelime haznesi boyutu: {len(self.vocab)}")

    def _add_special_tokens_to_vocab(self):
        """Özel tokenları kelime haznesine ekler."""
        current_id = 0
        for token in self.special_tokens:
            if token not in self.vocab:
                self.vocab[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
        # UNK token ID'sini kaydet
        self.unk_token_id = self.vocab[self.unk_token]

    def _build_vocab_from_chars(self, chars: Union[List[str], Set[str]]):
        """Verilen karakterlerden kelime haznesini oluşturur."""
        for char in chars:
            if char not in self.vocab:
                current_id = len(self.vocab) # Yeni ID mevcut kelime haznesinin boyutu olacak
                self.vocab[char] = current_id
                self.id_to_token[current_id] = char

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Metni karakter ID'lerine dönüştürür.
        """
        ids = []
        if add_special_tokens and BOS_TOKEN in self.vocab:
            ids.append(self.vocab[BOS_TOKEN])

        for char in text:
            ids.append(self.vocab.get(char, self.unk_token_id))

        if add_special_tokens and EOS_TOKEN in self.vocab:
            ids.append(self.vocab[EOS_TOKEN])
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Karakter ID'lerini metne dönüştürür.
        """
        chars = []
        for id_val in ids:
            token = self.id_to_token.get(id_val, self.unk_token)
            if skip_special_tokens and token in self.special_tokens:
                continue
            chars.append(token)
        return "".join(chars)

    def save_vocabulary(self, save_path: str):
        """
        Tokenleyici kelime haznesini bir JSON dosyasına kaydeder.
        """
        tokenizer_file = os.path.join(save_path, "char_vocab.json")
        os.makedirs(save_path, exist_ok=True)
        with open(tokenizer_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=4)
        log.info(f"CharTokenizer kelime haznesi '{tokenizer_file}' konumuna kaydedildi.")

    @classmethod
    def from_pretrained(cls, path: str) -> "CharTokenizer":
        """
        Kaydedilmiş bir kelime haznesinden CharTokenizer'ı yükler.
        """
        tokenizer_file = os.path.join(path, "char_vocab.json") if os.path.isdir(path) else path
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"CharTokenizer kelime haznesi dosyası bulunamadı: {tokenizer_file}")
        
        with open(tokenizer_file, "r", encoding="utf-8") as f:
            vocab_loaded = json.load(f)
        
        # Sadece karakterleri al, özel tokenları otomatik ekleyen init'i kullan.
        # Böylece ID'ler tutarlı kalır (özel tokenlar her zaman başta).
        chars_only = [char for char in vocab_loaded if char not in [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]]
        
        instance = cls(chars=chars_only, unk_token=UNK_TOKEN)
        # Yüklenen vocab'ı doğrudan atamak yerine, init metodumuzun oluşturduğu vocab'ı güncelleyelim.
        # Bu, özel tokenların ID'lerinin doğru sıralamada olmasını sağlar.
        # Bu kısım biraz karmaşıklaşabilir, basitleştirelim:
        
        # Basit yükleme: Eğer özel token ID'leri de kaydedildiyse, onları kullan.
        # Eğer özel token ID'leri dinamik olarak atandıysa, yeniden atamamız gerekebilir.
        # Şimdilik, yüklenen vocab'ın özel tokenları doğru ID'lerde içerdiğini varsayalım.
        instance.vocab = vocab_loaded
        instance.id_to_token = {v: k for k, v in vocab_loaded.items()}
        instance.unk_token_id = instance.vocab[UNK_TOKEN] # Yüklenen vocab'dan UNK ID'sini al
        
        log.info(f"CharTokenizer kelime haznesi '{tokenizer_file}' konumundan yüklendi. Boyut: {len(instance.vocab)}")
        return instance

    @property
    def vocabulary_size(self) -> int:
        """Kelime haznesi boyutunu döndürür."""
        return len(self.vocab)

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("CharTokenizer testi başlatılıyor...")

    # Loglama sistemini kur (test için gerekli)
    from src.utils.logger import setup_logging
    test_output_dir = "test_runs_char_tokenizer" # Klasör adını daha spesifik yaptık
    test_run_name = "char_tokenizer_test_run"
    
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

    # Test için karakterler
    test_chars = list("abcçdefgğhıijklmnoöprsştuüvyz ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ0123456789.,!?")

    # CharTokenizer oluştur
    print("\n--- CharTokenizer Oluşturma Testi ---")
    tokenizer = CharTokenizer(chars=test_chars)
    print(f"  Oluşturulan kelime haznesi boyutu: {tokenizer.vocabulary_size}")
    assert tokenizer.vocabulary_size > len(test_chars), "Kelime haznesi boyutu yanlış!" # Özel tokenlar da eklendiği için büyük olmalı
    print("CharTokenizer oluşturma testi başarılı. ✅")

    # Encode/Decode Testi
    print("\n--- Encode/Decode Testi ---")
    test_text = "Merhaba Dünya! 123"
    encoded_ids = tokenizer.encode(test_text, add_special_tokens=True)
    decoded_text = tokenizer.decode(encoded_ids, skip_special_tokens=True)
    
    print(f"  Orijinal Metin: '{test_text}'")
    print(f"  Kodlanmış ID'ler (özel tokenlarla): {encoded_ids}")
    print(f"  Çözülmüş Metin: '{decoded_text}'")
    
    assert decoded_text == test_text, "Encode/Decode tutarsızlığı!"
    print("Encode/Decode testi başarılı. ✅")

    # Bilinmeyen Token Testi
    print("\n--- Bilinmeyen Token Testi ---")
    unknown_text = "✨🚀🔥" # Kelime haznesinde olmayan karakterler
    encoded_unknown = tokenizer.encode(unknown_text, add_special_tokens=False)
    decoded_unknown = tokenizer.decode(encoded_unknown, skip_special_tokens=False) # Skip special tokens yapmayalım
    
    print(f"  Bilinmeyen Metin: '{unknown_text}'")
    print(f"  Kodlanmış Bilinmeyen ID'ler: {encoded_unknown}")
    print(f"  Çözülmüş Bilinmeyen Metin: '{decoded_unknown}'")
    
    # Tüm bilinmeyen karakterlerin UNK tokenına dönüştürüldüğünü kontrol et
    assert all(id_val == tokenizer.unk_token_id for id_val in encoded_unknown), "Bilinmeyen token eşleşmesi yanlış!"
    assert all(char == tokenizer.unk_token for char in decoded_unknown), "Bilinmeyen karakter çözülürken yanlış!"
    print("Bilinmeyen token testi başarılı. ✅")

    # Kaydetme ve Yükleme Testi
    print("\n--- Kaydetme ve Yükleme Testi ---")
    tokenizer_save_path = os.path.join(test_output_dir, "char_tokenizer_assets")
    tokenizer.save_vocabulary(tokenizer_save_path)
    
    loaded_tokenizer = CharTokenizer.from_pretrained(tokenizer_save_path)
    assert loaded_tokenizer.vocabulary_size == tokenizer.vocabulary_size, "Yüklenen kelime haznesi boyutu yanlış!"
    assert loaded_tokenizer.encode("test", add_special_tokens=False) == tokenizer.encode("test", add_special_tokens=False), "Yüklenen tokenleyici tutarsız!"
    print("Kaydetme ve yükleme testi başarılı. ✅")

    print("\nCharTokenizer tüm testleri tamamlandı. ✅")

    # Test sonrası oluşturulan test dizinini temizle (finally bloğunda olmalıydı ama burada manuel kontrol)
    # Bu kısmı ana test betiğindeki finally bloğuna taşıdık.
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
        print(f"Temizlik yapıldı: '{test_output_dir}' silindi.")