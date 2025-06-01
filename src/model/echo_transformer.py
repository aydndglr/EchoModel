
# src/model/echo_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from src.config.base_config import BaseConfig
from src.config.model_config import ModelConfig
from src.model.components.embeddings import EmbeddingLayer # Gömme katmanımız
from src.model.components.layer_norm import LayerNorm # Son katman normalizasyonu
from src.model.transformer_decoder_block import TransformerDecoderBlock # Decoder bloğumuz

class EchoTransformer(nn.Module):
    """
    EchoModel'in ana Decoder-Only Transformer modeli.
    Gömme katmanını, birden çok Transformer Decoder bloğunu ve bir çıkış katmanını içerir.
    """
    def __init__(self, base_config: BaseConfig, model_config: ModelConfig):
        super().__init__()
        self.base_config = base_config
        self.model_config = model_config
        
        self.vocab_size = base_config.vocab_size
        self.d_model = model_config.d_model
        self.n_layers = model_config.n_layers
        
        # OLMO'daki `transformer` ModuleDict'ine benzer bir yapı kuruyoruz.
        # Token gömmeleri ve konumsal gömmeleri içeren ana gömme katmanı
        self.embeddings = EmbeddingLayer(self.vocab_size, model_config, base_config)
        
        # Transformer Decoder Blokları
        # OLMO'da bu `nn.ModuleList` veya `OLMoBlockGroup` içinde yer alır.
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(model_config, base_config)
            for _ in range(self.n_layers)
        ])
        
        # Son Layer Normalizasyon katmanı (genellikle Transformer çıktıdan sonra)
        # OLMO'da `ln_f` olarak geçer.
        self.final_norm = LayerNorm(self.d_model, 
                                    eps=base_config.adam_epsilon, 
                                    use_rms_norm=model_config.use_rms_norm, 
                                    bias=False) # RMSNorm'da bias False

        # Çıkış doğrusal katmanı (logits üretir)
        # OLMO'da `ff_out` olarak geçebilir veya `weight_tying` varsa `wte.weight` ile paylaşılır.
        # Biz şimdilik ayrı bir lineer katman olarak tutalım.
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=True) # Bias True varsayalım

        # Ağırlık paylaşımı (Weight Tying) opsiyonu
        # Genellikle LM Head'in ağırlıkları, token gömmelerinin ağırlıklarıyla paylaşılır.
        # Bu, model boyutunu küçültür ve performansı artırabilir.
        # OLMO'da `weight_tying` config'ten kontrol edilir.
        if getattr(model_config, 'weight_tying', True): # Varsayılan olarak True alalım
            self.lm_head.weight = self.embeddings.token_embeddings.word_embeddings.weight
            # Weight tying kullanırken bias'ı da paylaşmak istersen ayrı bir mekanizma gerekebilir.
            # Şimdilik sadece ağırlıkları paylaşalım.

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            input_ids (torch.LongTensor): Giriş token ID'leri (batch_size, seq_len).
            attention_mask (Optional[torch.Tensor]): Padding maskesi (batch_size, seq_len).
            past_key_values (Optional[List[Tuple[torch.Tensor, torch.Tensor]]]): Önceki K/V değerleri listesi.
            use_cache (bool): K/V değerlerini döndürüp döndürmeyeceği.

        Returns:
            Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
                - Logits (batch_size, seq_len, vocab_size).
                - present_key_values (liste halinde K, V tuple'ları), use_cache True ise.
        """
        # Gömme katmanı
        # input_ids (B, T) -> x (B, T, D)
        x = self.embeddings(input_ids)

        # Cache kullanımında past_key_values'ın uzunluğunu kontrol et
        if past_key_values:
            assert len(past_key_values) == self.n_layers, \
                f"past_key_values uzunluğu ({len(past_key_values)}) n_layers ({self.n_layers}) ile eşleşmeli."

        # Transformer Decoder Blokları
        present_key_values_list = [] if use_cache else None
        
        for i, block in enumerate(self.decoder_blocks):
            layer_past = None if past_key_values is None else past_key_values[i]
            
            # Her bir bloğu çağır
            x, present_kv_block = block(
                x,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=use_cache
            )
            
            if present_key_values_list is not None:
                present_key_values_list.append(present_kv_block)

        # Son normalizasyon
        x = self.final_norm(x)

        # Logits katmanı (d_model -> vocab_size)
        # shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits, present_key_values_list

    def num_params(self) -> int:
        """
        Modeldeki toplam parametre sayısını döndürür.
        Weight tying varsa lm_head parametrelerini tekrar saymaz.
        """
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Eğer weight_tying varsa lm_head'in ağırlıkları zaten embeddings içinde sayılır
        # lm_head'in bias'ı varsa o sayılır.
        if getattr(self.model_config, 'weight_tying', True) and self.lm_head.bias is not None:
             # lm_head.weight zaten sayıldı, sadece bias'ı ekle
             num_parameters_without_tied_weight = sum(p.numel() for name, p in self.named_parameters() 
                                                       if p.requires_grad and "lm_head.weight" not in name)
             return num_parameters_without_tied_weight
        return num_parameters


# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("EchoTransformer (Ana Model) katmanı testi başlatılıyor...")

    from src.config.base_config import BaseConfig
    from src.config.model_config import ModelConfig
    
    base_cfg = BaseConfig()
    model_cfg = ModelConfig()

    # Model oluştur
    model = EchoTransformer(base_config=base_cfg, model_config=model_cfg)
    print(f"EchoTransformer modeli başarıyla oluşturuldu. Toplam parametre: {model.num_params() / 1e6:.2f} Milyon.")
    
    # Modelin doğru cihazda olup olmadığını kontrol et (eğer cuda varsa)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model cihazda: {next(model.parameters()).device}")

    # Test girdileri
    batch_size = 2
    seq_len = 16 # Test için daha kısa bir sekans
    dummy_input_ids = torch.randint(0, base_cfg.vocab_size, (batch_size, seq_len)).to(device)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    
    # Test Senaryosu 1: Temel ileri besleme (cache'siz)
    print("\n--- Test Senaryosu 1: Temel İleri Besleme (Cache'siz) ---")
    logits_s1, present_kv_s1 = model(dummy_input_ids, attention_mask=dummy_attention_mask, use_cache=False)
    print(f"  Logits boyutu: {logits_s1.shape} (Beklenen: {batch_size, seq_len, base_cfg.vocab_size})")
    assert logits_s1.shape == (batch_size, seq_len, base_cfg.vocab_size), "  S1 logits boyutu yanlış!"
    assert present_kv_s1 is None, "  S1 Cache beklenmedi ama döndürüldü!"
    print("  Test Senaryosu 1 başarılı. ✅")

    # Test Senaryosu 2: Cache kullanımı (metin üretimi modu)
    print("\n--- Test Senaryosu 2: Cache Kullanımı (Metin Üretimi Modu) ---")
    # İlk token için forward
    first_token_input_ids = torch.randint(0, base_cfg.vocab_size, (batch_size, 1)).to(device)
    first_token_attention_mask = torch.ones(batch_size, 1, dtype=torch.bool).to(device)
    
    logits_step1, present_kv_step1 = model(first_token_input_ids, attention_mask=first_token_attention_mask, use_cache=True)
    print(f"  Adım 1 Logits boyutu: {logits_step1.shape}")
    assert logits_step1.shape == (batch_size, 1, base_cfg.vocab_size), "  Adım 1 logits boyutu yanlış!"
    assert present_kv_step1 is not None, "  Adım 1 Cache bekleniyordu ama döndürülmedi!"
    assert len(present_kv_step1) == model_cfg.n_layers, "  Adım 1 Cache listesi uzunluğu yanlış!"
    print("  Adım 1 başarılı. ✅")

    # İkinci token için forward (önceki cache ile)
    second_token_input_ids = torch.randint(0, base_cfg.vocab_size, (batch_size, 1)).to(device)
    # Maske de uzamalı
    second_token_attention_mask = torch.ones(batch_size, 2, dtype=torch.bool).to(device) # Total seq_len = 2

    logits_step2, present_kv_step2 = model(
        second_token_input_ids,
        attention_mask=second_token_attention_mask,
        past_key_values=present_kv_step1, # Önceki cache'i buraya verdik
        use_cache=True
    )
    print(f"  Adım 2 Logits boyutu: {logits_step2.shape}")
    assert logits_step2.shape == (batch_size, 1, base_cfg.vocab_size), "  Adım 2 logits boyutu yanlış!"
    assert present_kv_step2 is not None, "  Adım 2 Cache bekleniyordu ama döndürülmedi!"
    assert len(present_kv_step2) == model_cfg.n_layers, "  Adım 2 Cache listesi uzunluğu yanlış!"
    # Her bir bloğun cache boyutu (batch_size, n_heads, current_seq_len, head_dim) olmalı
    # current_seq_len artık 2 olmalı
    assert present_kv_step2[0][0].shape[-2] == 2, "  Adım 2 Cache boyutu yanlış!"
    print("  Adım 2 başarılı. ✅")

    print("\nEchoTransformer (Ana Model) katmanı tüm testleri tamamlandı. ✅")