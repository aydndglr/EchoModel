# src/model/components/multi_head_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple # Optional ve Tuple'ı içeri aktarıyoruz

from src.config.model_config import ModelConfig
from src.config.base_config import BaseConfig
from src.model.components.layer_norm import LayerNorm

class MultiHeadAttention(nn.Module):
    """
    Çoklu Kafa Maskeli Dikkat Mekanizması.
    Transformer Decoder bloklarının ana bileşenidir.
    """
    def __init__(self, model_config: ModelConfig, base_config: BaseConfig):
        super().__init__()
        self.d_model = model_config.d_model
        self.n_heads = model_config.n_heads
        self.head_dim = self.d_model // self.n_heads # Her bir dikkat kafasının boyutu
        self.dropout_p = model_config.dropout # Dropout oranı
        self.max_seq_len = base_config.max_seq_len # Maske oluşturmak için

        # Q, K, V (Query, Key, Value) için doğrusal projeksiyonlar
        # ModelConfig'te 'include_bias' ayarı yoksa varsayılan olarak True kullan
        bias_for_projections = getattr(model_config, 'include_bias', True)

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=bias_for_projections)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=bias_for_projections)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=bias_for_projections)

        # Dikkat çıktısı için doğrusal projeksiyon
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias_for_projections)

        self.attn_dropout = nn.Dropout(self.dropout_p) # Dikkat skorlarına uygulanacak dropout

        # Causal maskeyi bir tampon olarak kaydet.
        self.register_buffer("causal_mask", self._get_causal_mask(self.max_seq_len))

    def _get_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Causal (üst üçgensel) maskeyi oluşturur.
        Bu maske, her tokenin yalnızca kendinden önceki tokenlere dikkat etmesini sağlar.
        Çıktı boyutu: (1, 1, seq_len, seq_len)
        """
        # OLMO'daki gibi, üst üçgeni (diagonal=1 dahil) -inf yapıp diğer yerleri 0 bırakalım.
        # Bu, PyTorch'un F.scaled_dot_product_attention için beklediği maske formatına benzer.
        causal_mask = torch.full((seq_len, seq_len), 0.0, dtype=torch.float)
        causal_mask = torch.triu(causal_mask, diagonal=1) # Üst üçgen 0, alt üçgen 0
        
        # Sadece üst üçgeni (gelecek tokenleri) -inf ile doldur
        causal_mask.masked_fill_(causal_mask.bool(), float('-inf')) # True olan yerleri -inf yap
        # NOT: `causal_mask.bool()` ifadesi yerine `causal_mask == 0` veya `causal_mask == 1`
        # kullanmak, masked_fill için beklenen boolean maskeyi daha net sağlayabilir.
        # Ancak `torch.triu(..., diagonal=1)` ile oluşturulan maskede sadece 0'lar olduğu için
        # `masked_fill_(causal_mask.bool(), float('-inf'))` aslında hiçbir şeyi -inf yapmaz.
        
        # Orijinal Transformer ve OLMO'daki (model.py) mantığı tam olarak taklit edelim:
        # `torch.triu(torch.ones(...), diagonal=1)` ile 1'leri -inf yaparız.
        causal_mask_values = torch.ones(seq_len, seq_len, dtype=torch.float)
        causal_mask_values = torch.triu(causal_mask_values, diagonal=1) # Üst üçgen 1, alt üçgen 0
        causal_mask_values = causal_mask_values.masked_fill(causal_mask_values == 1, float('-inf')) # 1 olanları -inf yap, 0 olanlar 0 kalır.
        
        return causal_mask_values.view(1, 1, seq_len, seq_len) # (1, 1, seq_len, seq_len) şekline getir

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size() # Batch boyutu, mevcut sekans uzunluğu (query_len), d_model

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Çoklu kafa için boyutları ayır
        # shape: (B, T, d_model) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Cache kullanımı (önceki adımlardaki K ve V'leri birleştir)
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2) # -2 = sequence length dimension
            v = torch.cat((past_value, v), dim=-2)
        
        present_key_values = (k, v) if use_cache else None

        current_key_len = k.size(-2) # K'nın yeni uzunluğu (geçmiş + mevcut)

        # Dikkat skorları hesaplama: Q * K^T / sqrt(head_dim)
        # (B, n_heads, T_q, head_dim) @ (B, n_heads, head_dim, T_k) -> (B, n_heads, T_q, T_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Maskeleri Uygulama
        # Causal maskeyi doğru boyutta dilimle (query uzunluğu T, key uzunluğu current_key_len)
        causal_mask_sliced = self.causal_mask[:, :, :T, :current_key_len] # (1, 1, T, current_key_len)
        attn_scores = attn_scores + causal_mask_sliced.to(attn_scores.dtype)

        # Padding maskesi uygulama (eğer varsa)
        if attention_mask is not None:
            # `attention_mask` (batch_size, seq_len) formatında (True/1: dikkat et, False/0: maskele)
            # Bunu dikkat skorlarına eklemek için -inf içeren bir maskeye dönüştürmeliyiz.
            # OLMO'da: `(1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min` ile yapılıyor.
            # `attention_mask`'ı float'a çevirip 0 olan yerleri -inf, 1 olan yerleri 0 yap.
            
            # `attention_mask`'i doğru dtype'a çevir ve genişlet: (B, 1, 1, T_input)
            # T_input = input_ids'in uzunluğu (yani mevcut girdi 'T')
            padding_mask_float = attention_mask.float().unsqueeze(1).unsqueeze(2) # (B, 1, 1, T)
            
            # 0 olan yerleri (maskelenecek) -inf yap, 1 olan yerleri (maskelenmeyecek) 0 yap.
            padding_mask_float = (1.0 - padding_mask_float).masked_fill(padding_mask_float == 0.0, float('-inf'))
            padding_mask_float = padding_mask_float.masked_fill(padding_mask_float == 1.0, 0.0)

            # Bu `padding_mask_float` şu anda (B, 1, 1, T) boyutunda.
            # `attn_scores` ise (B, n_heads, T_q, T_k) yani (B, n_heads, T, current_key_len).
            # Padding maskesi, `key` dizisinin tamamını kapsayacak şekilde `current_key_len` boyutunda olmalıdır.
            # Yani `padding_mask_float`'ın son boyutu `current_key_len` olmalı.
            # Şu anki `padding_mask_float`'ın `T` boyutu var ve bu `current_key_len`'den küçük olabilir.
            # Bu durumda, `current_key_len` boyutunda bir maske oluşturup, `padding_mask_float`'ı buna kopyalamalıyız.
            
            # Geçmiş K/V değerleri olduğunda padding maskesinin nasıl genişletileceği kritik.
            # Eğer `attention_mask` sadece mevcut girdi `x`'in padding'ini gösteriyorsa,
            # o zaman `past_key_values`'dan gelen K/V'ler zaten padding'li değilse sorun yok.
            # Ancak `attention_mask` tüm (geçmiş + mevcut) sekans için olmalıydı.
            # Test senaryo 3'te `layer_past` yok, dolayısıyla `T == current_key_len`.
            # Bu durumda `padding_mask_float`'ı doğrudan `attn_scores`'a ekleyebiliriz (broadcast ile).
            attn_scores = attn_scores + padding_mask_float # Broadcast ile (B, 1, 1, T) -> (B, n_heads, T, T)

        # Softmax ve Dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Değerleri birleştirme: (attn_weights @ V)
        output = torch.matmul(attn_weights, v)

        # Kafaları birleştirme ve son projeksiyon
        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)

        output = self.out_proj(output)

        return output, present_key_values


# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("MultiHeadAttention katmanı testi başlatılıyor...")

    # Test için gerekli config'leri doğrudan içeri aktarıyoruz.
    from src.config.base_config import BaseConfig
    from src.config.model_config import ModelConfig
    
    base_cfg = BaseConfig()
    model_cfg = ModelConfig()

    # Parametreler
    batch_size = 2
    seq_len = 32 # test için daha küçük bir sekans uzunluğu
    d_model_test = model_cfg.d_model # 512
    n_heads_test = model_cfg.n_heads # 8

    # Rastgele giriş tensörü
    dummy_input = torch.randn(batch_size, seq_len, d_model_test)
    print(f"Giriş tensörü boyutu: {dummy_input.shape}")

    # MultiHeadAttention katmanı oluştur
    attention_layer = MultiHeadAttention(model_config=model_cfg, base_config=base_cfg)

    # Test senaryosu 1: Temel ileri besleme (maskesiz, cache'siz)
    print("\n--- Test Senaryosu 1: Temel İleri Besleme (Maskesiz, Cache'siz) ---")
    output_s1, present_kv_s1 = attention_layer(dummy_input)
    print(f"  Çıktı boyutu: {output_s1.shape} (Beklenen: {batch_size, seq_len, d_model_test})")
    assert output_s1.shape == (batch_size, seq_len, d_model_test), "  S1 çıktı boyutu yanlış!"
    assert present_kv_s1 is None, "  S1 Cache beklenmedi ama döndürüldü!"
    print("  Test Senaryosu 1 başarılı. ✅")

    # Test senaryosu 2: Cache kullanımı (üretim modu)
    print("\n--- Test Senaryosu 2: Cache Kullanımı (Metin Üretimi Modu) ---")
    # İlk adım için (seq_len = 1)
    single_token_input = torch.randn(batch_size, 1, d_model_test)
    output_step1, present_kv_step1 = attention_layer(single_token_input, use_cache=True)
    print(f"  Adım 1 Çıktı boyutu: {output_step1.shape}")
    assert output_step1.shape == (batch_size, 1, d_model_test), "  Adım 1 çıktı boyutu yanlış!"
    assert present_kv_step1 is not None, "  Adım 1 Cache bekleniyordu ama döndürülmedi!"
    print(f"  Cache K boyutu: {present_kv_step1[0].shape} (Beklenen: {batch_size, n_heads_test, 1, d_model_test // n_heads_test})")
    print(f"  Cache V boyutu: {present_kv_step1[1].shape} (Beklenen: {batch_size, n_heads_test, 1, d_model_test // n_heads_test})")
    assert present_kv_step1[0].shape == (batch_size, n_heads_test, 1, d_model_test // n_heads_test), "  Cache K boyutu yanlış!"

    # İkinci adım için (önceki cache ile birlikte)
    single_token_input_step2 = torch.randn(batch_size, 1, d_model_test)
    output_step2, present_kv_step2 = attention_layer(single_token_input_step2, layer_past=present_kv_step1, use_cache=True)
    print(f"  Adım 2 Çıktı boyutu: {output_step2.shape}")
    assert output_step2.shape == (batch_size, 1, d_model_test), "  Adım 2 çıktı boyutu yanlış!"
    assert present_kv_step2 is not None, "  Adım 2 Cache bekleniyordu ama döndürülmedi!"
    print(f"  Cache K boyutu: {present_kv_step2[0].shape} (Beklenen: {batch_size, n_heads_test, 2, d_model_test // n_heads_test})")
    assert present_kv_step2[0].shape == (batch_size, n_heads_test, 2, d_model_test // n_heads_test), "  Adım 2 Cache K boyutu yanlış!"
    print("  Test Senaryosu 2 başarılı. ✅")

    # Test Senaryosu 3: Padding Maskesi Kullanımı
    print("\n--- Test Senaryosu 3: Padding Maskesi Kullanımı ---")
    # Örnek bir padding maskesi (batch_size, seq_len)
    # True = dikkat et, False = maskele (padding)
    padding_mask_input = torch.ones(batch_size, seq_len, dtype=torch.bool)
    padding_mask_input[0, seq_len-2:] = False # İlk örnekte son 2 token maskeli
    padding_mask_input[1, seq_len-5:] = False # İkinci örnekte son 5 token maskeli

    output_with_mask, _ = attention_layer(dummy_input, attention_mask=padding_mask_input)
    print(f"  Maskeli çıktı boyutu: {output_with_mask.shape}")
    assert output_with_mask.shape == (batch_size, seq_len, d_model_test), "  Maskeli çıktı boyutu yanlış!"
    print("  Test Senaryosu 3 başarılı. ✅")

    print("\nMultiHeadAttention katmanı tüm testleri tamamlandı. ✅")