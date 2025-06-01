
# src/model/transformer_decoder_block.py

import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.config.model_config import ModelConfig
from src.config.base_config import BaseConfig
from src.model.components.layer_norm import LayerNorm # Kendi LayerNorm'umuzu kullanıyoruz
from src.model.components.multi_head_attention import MultiHeadAttention # Kendi MultiHeadAttention'ımızı kullanıyoruz
from src.model.components.feed_forward import FeedForward # Kendi FeedForward'ımızı kullanıyoruz

class TransformerDecoderBlock(nn.Module):
    """
    Tek bir Transformer Decoder Bloğu.
    Attention ve Feed-Forward katmanlarını bir araya getirir.
    Pre-normalization (normların dikkat/FFN'den önce uygulanması) ve kalıntı bağlantıları içerir.
    """
    def __init__(self, model_config: ModelConfig, base_config: BaseConfig):
        super().__init__()
        self.d_model = model_config.d_model
        self.dropout_p = model_config.dropout
        self.pre_normalization = model_config.pre_normalization
        self.use_rms_norm = model_config.use_rms_norm
        
        # OLMO'da bu değerler config'ten geliyor.
        self.eps = base_config.adam_epsilon # LayerNorm için epsilon

        # Normalizasyon katmanları
        # OLMO'da pre-normalization kullanıldığında, her katmandan önce LayerNorm bulunur.
        self.attn_norm = LayerNorm(self.d_model, eps=self.eps, use_rms_norm=self.use_rms_norm, bias=False) # RMSNorm'da bias genellikle False
        self.ff_norm = LayerNorm(self.d_model, eps=self.eps, use_rms_norm=self.use_rms_norm, bias=False)

        # Dikkat Mekanizması
        self.attention = MultiHeadAttention(model_config, base_config)
        self.attn_dropout = nn.Dropout(self.dropout_p) # Dikkat katmanından sonraki residual dropout

        # İleri Besleme Ağı
        self.feed_forward = FeedForward(model_config)
        self.ff_dropout = nn.Dropout(self.dropout_p) # FFN katmanından sonraki residual dropout

        # Kalıntı bağlantıları (residual connections)
        # Dropout katmanları modülün içinde tanımlandığı için burada sadece toplama yaparız.
        # OLMO'da `residual_dropout` veya `resid_p` gibi isimlerle genel dropout'tan ayrı yönetilir.
        # Bizim ModelConfig'imizde tek bir `dropout` var, onu kullanıyoruz.

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x (torch.Tensor): Giriş tensörü (batch_size, seq_len, d_model).
            attention_mask (Optional[torch.Tensor]): Padding maskesi.
            layer_past (Optional[Tuple[torch.Tensor, torch.Tensor]]): Önceki K ve V değerleri.
            use_cache (bool): K ve V değerlerini döndürüp döndürmeyeceği.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                - Bloğun çıktısı (batch_size, seq_len, d_model).
                - present_key_values (key, value) tuple'ı (sadece use_cache True ise).
        """
        
        # --- Birinci Alt Blok: Dikkat Mekanizması ---
        # OLMO'da `norm_after` konfigürasyonu var. Biz `pre_normalization=True` olarak modelledik.
        # Yani önce normalizasyon, sonra dikkat.
        
        # Kalıntı bağlantısı için orijinal girdiyi sakla
        residual = x

        # Ön-normalizasyon (pre-normalization)
        x_normed = self.attn_norm(x)

        # Dikkat katmanı
        attn_output, present_key_values = self.attention(
            x_normed, # Normlanmış girdiyi kullan
            attention_mask=attention_mask,
            layer_past=layer_past,
            use_cache=use_cache
        )
        
        # Dropout ve kalıntı bağlantısı
        x = residual + self.attn_dropout(attn_output) # Normlanmamış x ile topluyoruz


        # --- İkinci Alt Blok: İleri Besleme Ağı ---
        # Kalıntı bağlantısı için mevcut çıktıyı sakla
        residual = x

        # Ön-normalizasyon (pre-normalization)
        x_normed = self.ff_norm(x)

        # İleri besleme ağı
        ff_output = self.feed_forward(x_normed) # Normlanmış girdiyi kullan

        # Dropout ve kalıntı bağlantısı
        x = residual + self.ff_dropout(ff_output) # Normlanmamış x ile topluyoruz

        return x, present_key_values

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("TransformerDecoderBlock katmanı testi başlatılıyor...")

    # Test için gerekli config'leri doğrudan içeri aktarıyoruz.
    from src.config.base_config import BaseConfig
    from src.config.model_config import ModelConfig
    
    base_cfg = BaseConfig()
    model_cfg = ModelConfig()

    # Parametreler
    batch_size = 2
    seq_len = 32 # Test için daha küçük bir sekans uzunluğu
    d_model_test = model_cfg.d_model # 512

    # Rastgele giriş tensörü
    dummy_input = torch.randn(batch_size, seq_len, d_model_test)
    print(f"Giriş tensörü boyutu: {dummy_input.shape}")

    # TransformerDecoderBlock katmanı oluştur
    decoder_block = TransformerDecoderBlock(model_config=model_cfg, base_config=base_cfg)

    # Test senaryosu 1: Temel ileri besleme (maskesiz, cache'siz)
    print("\n--- Test Senaryosu 1: Temel İleri Besleme (Maskesiz, Cache'siz) ---")
    output_s1, present_kv_s1 = decoder_block(dummy_input)
    print(f"  Çıktı boyutu: {output_s1.shape} (Beklenen: {batch_size, seq_len, d_model_test})")
    assert output_s1.shape == (batch_size, seq_len, d_model_test), "  S1 çıktı boyutu yanlış!"
    assert present_kv_s1 is None, "  S1 Cache beklenmedi ama döndürüldü!"
    print("  Test Senaryosu 1 başarılı. ✅")

    # Test senaryosu 2: Cache kullanımı (üretim modu)
    print("\n--- Test Senaryosu 2: Cache Kullanımı (Metin Üretimi Modu) ---")
    single_token_input = torch.randn(batch_size, 1, d_model_test) # Tek tokenlık girdi
    output_step1, present_kv_step1 = decoder_block(single_token_input, use_cache=True)
    print(f"  Adım 1 Çıktı boyutu: {output_step1.shape}")
    assert output_step1.shape == (batch_size, 1, d_model_test), "  Adım 1 çıktı boyutu yanlış!"
    assert present_kv_step1 is not None, "  Adım 1 Cache bekleniyordu ama döndürülmedi!"

    # İkinci adım için (önceki cache ile birlikte)
    single_token_input_step2 = torch.randn(batch_size, 1, d_model_test)
    output_step2, present_kv_step2 = decoder_block(single_token_input_step2, layer_past=present_kv_step1, use_cache=True)
    print(f"  Adım 2 Çıktı boyutu: {output_step2.shape}")
    assert output_step2.shape == (batch_size, 1, d_model_test), "  Adım 2 çıktı boyutu yanlış!"
    assert present_kv_step2 is not None, "  Adım 2 Cache bekleniyordu ama döndürülmedi!"
    print(f"  Adım 2 Cache K boyutu: {present_kv_step2[0].shape} (Beklenen: {batch_size, model_cfg.n_heads, 2, model_cfg.d_model // model_cfg.n_heads})")
    print("  Test Senaryosu 2 başarılı. ✅")

    # Test Senaryosu 3: Padding Maskesi Kullanımı
    print("\n--- Test Senaryosu 3: Padding Maskesi Kullanımı ---")
    # Örnek bir padding maskesi (batch_size, seq_len)
    padding_mask_input = torch.ones(batch_size, seq_len, dtype=torch.bool)
    padding_mask_input[0, seq_len-2:] = False # İlk örnekte son 2 token maskeli
    padding_mask_input[1, seq_len-5:] = False # İkinci örnekte son 5 token maskeli

    output_with_mask, _ = decoder_block(dummy_input, attention_mask=padding_mask_input)
    print(f"  Maskeli çıktı boyutu: {output_with_mask.shape}")
    assert output_with_mask.shape == (batch_size, seq_len, d_model_test), "  Maskeli çıktı boyutu yanlış!"
    print("  Test Senaryosu 3 başarılı. ✅")

    print("\nTransformerDecoderBlock katmanı tüm testleri tamamlandı. ✅")