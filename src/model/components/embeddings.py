# src/model/components/embeddings.py

import torch
import torch.nn as nn
# BaseConfig ve ModelConfig'i içeri aktarıyoruz.
from src.config.base_config import BaseConfig
from src.config.model_config import ModelConfig

class TokenEmbeddings(nn.Module):
    """
    Modelin kelime haznesindeki her bir tokene ait öğrenilebilir gömme vektörlerini tanımlar.
    Bu vektörler, tokenların anlamsal temsillerini içerir.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # nn.Embedding: Giriş olarak token ID'lerini (long tipinde) alır ve
        # d_model boyutunda yoğun vektörlere dönüştürür.
        self.word_embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids (torch.Tensor): Token ID'lerini içeren tensör (batch_size, seq_len).

        Returns:
            torch.Tensor: Gömme vektörleri (batch_size, seq_len, d_model).
        """
        return self.word_embeddings(input_ids)

class PositionalEmbeddings(nn.Module):
    """
    Tokenların sırasını ve konumunu modele bildirmek için konumsal gömmeler ekler.
    Transformer'lar doğal olarak sıra bilgisini işlemez.
    Burada sinüzoidal (sabit) konumsal gömmeler kullanılır, tıpkı orijinal Transformer makalesinde olduğu gibi.
    """
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Sinüzoidal konumsal gömmeleri hesapla ve bir tampon olarak kaydet.
        # Bu, modelin parametreleri olarak öğrenilmez, sabittir.
        pe = torch.zeros(max_seq_len, d_model) # (max_seq_len, d_model) boyutunda sıfırlardan oluşur
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) # (max_seq_len, 1)
        
        # frekansları hesapla: 1 / (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # Çift indekslere sinüs uygula
        pe[:, 1::2] = torch.cos(position * div_term) # Tek indekslere kosinüs uygula
        
        pe = pe.unsqueeze(0) # (1, max_seq_len, d_model) -> batch boyutu için
        self.register_buffer('pe', pe) # Model durumu ile birlikte kaydedilir, ancak öğrenilmez.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Giriş tensörüne konumsal gömmeleri ekler.
        Args:
            x (torch.Tensor): Gömme vektörleri (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Konumsal bilgi eklenmiş gömme vektörleri (batch_size, seq_len, d_model).
        """
        # x'in mevcut sekans uzunluğuna göre konumsal gömmeyi dilimle
        # Bu, max_seq_len'den daha kısa diziler için uyumluluk sağlar.
        return x + self.pe[:, :x.size(1)]

class EmbeddingLayer(nn.Module):
    """
    Token ve konumsal gömmeleri birleştirir ve dropout uygular.
    """
    # __init__ metoduna base_config'i ekledik
    def __init__(self, vocab_size: int, model_config: ModelConfig, base_config: BaseConfig):
        super().__init__()
        self.token_embeddings = TokenEmbeddings(vocab_size, model_config.d_model)
        # max_seq_len'i base_config'ten alıyoruz
        self.positional_embeddings = PositionalEmbeddings(model_config.d_model, base_config.max_seq_len)
        self.dropout = nn.Dropout(model_config.embedding_dropout)
        self.d_model = model_config.d_model # Çıktıyı model boyutuna göre ölçeklendirmek için

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids (torch.Tensor): Token ID'lerini içeren tensör (batch_size, seq_len).

        Returns:
            torch.Tensor: Dropout uygulanmış, birleştirilmiş gömme vektörleri (batch_size, seq_len, d_model).
        """
        # Token gömmelerini al
        token_embeds = self.token_embeddings(input_ids)
        
        # Ölçeklendirme: Orijinal Transformer makalesinde gömmeler sqrt(d_model) ile çarpılır.
        # Bu, gömmelerin değer aralığını stabilize etmeye yardımcı olur.
        token_embeds = token_embeds * torch.sqrt(torch.tensor(self.d_model, dtype=token_embeds.dtype))
        
        # Konumsal gömmeleri ekle
        x = self.positional_embeddings(token_embeds)
        
        # Dropout uygula
        x = self.dropout(x)
        return x

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("Embeddings katmanları testi başlatılıyor...")

    # Örnek ModelConfig ve BaseConfig
    # Test için gerekli config'leri doğrudan içeri aktarıyoruz.
    from src.config.base_config import BaseConfig
    from src.config.model_config import ModelConfig
    base_cfg = BaseConfig() # Varsayılan değerlerle
    model_cfg = ModelConfig() # Varsayılan değerlerle

    # Parametreler
    _vocab_size = base_cfg.vocab_size # Örn: 50257
    _d_model = model_cfg.d_model # Örn: 512
    _max_seq_len = base_cfg.max_seq_len # Örn: 512

    # Rastgele giriş token ID'leri
    batch_size = 2
    seq_len = 128 # max_seq_len'den küçük olabilir
    dummy_input_ids = torch.randint(0, _vocab_size, (batch_size, seq_len))
    print(f"Giriş token ID'leri boyutu: {dummy_input_ids.shape}")

    # TokenEmbeddings testi
    token_embed_test = TokenEmbeddings(_vocab_size, _d_model)
    token_output = token_embed_test(dummy_input_ids)
    print(f"TokenEmbeddings çıktı boyutu: {token_output.shape} (Beklenen: {batch_size, seq_len, _d_model})")
    assert token_output.shape == (batch_size, seq_len, _d_model), "TokenEmbeddings boyutu yanlış!"
    print("TokenEmbeddings testi başarılı.")

    # PositionalEmbeddings testi
    pos_embed_test = PositionalEmbeddings(_d_model, _max_seq_len)
    # Token embedding'den gelen rastgele bir tensör üzerinde test edelim
    dummy_input_for_pos = torch.randn(batch_size, seq_len, _d_model)
    pos_output = pos_embed_test(dummy_input_for_pos)
    print(f"PositionalEmbeddings çıktı boyutu: {pos_output.shape} (Beklenen: {batch_size, seq_len, _d_model})")
    assert pos_output.shape == (batch_size, seq_len, _d_model), "PositionalEmbeddings boyutu yanlış!"
    # Konumsal gömme eklenip eklenmediğini kontrol etmek için küçük bir karşılaştırma
    assert not torch.equal(dummy_input_for_pos, pos_output), "Konumsal gömme eklenmedi!"
    print("PositionalEmbeddings testi başarılı.")

    # EmbeddingLayer (Genel) testi
    # BaseConfig objesini de parametre olarak geçirmeliyiz
    embedding_layer = EmbeddingLayer(_vocab_size, model_cfg, base_cfg) # Buradaki parametreleri düzelttik
    embedding_output = embedding_layer(dummy_input_ids)
    print(f"EmbeddingLayer çıktı boyutu: {embedding_output.shape} (Beklenen: {batch_size, seq_len, _d_model})")
    assert embedding_output.shape == (batch_size, seq_len, _d_model), "EmbeddingLayer boyutu yanlış!"
    print("EmbeddingLayer testi başarılı.")

    print("\nEmbeddings katmanları tüm testleri tamamlandı.")