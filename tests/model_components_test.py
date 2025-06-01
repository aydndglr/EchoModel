# EchoModel/tests/model_components_test.py

import os
import sys
import torch
import torch.nn as nn
import logging
import shutil # Sadece import kalsın, rmtree çağrısı kaldırıldı

# --- Proje Kök Dizinini Bulma ve PYTHONPATH Ayarı ---
def find_project_root(current_path: str) -> str:
    """
    Geçerli dizinden yukarı doğru çıkarak 'src' klasörünü içeren proje kök dizinini bulur.
    """
    temp_path = os.path.abspath(current_path)
    while temp_path != os.path.dirname(temp_path):
        if os.path.exists(os.path.join(temp_path, 'src', '__init__.py')):
            return temp_path
        temp_path = os.path.dirname(temp_path)
    raise RuntimeError("EchoModel proje kök dizini ('src' klasörü içeren) bulunamadı. Lütfen dizin yapısını veya betiğin konumunu kontrol edin.")

# Betiğin çalıştığı dizinden proje kökünü bul
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = find_project_root(current_script_dir)

# Proje kök dizinini Python yoluna ekle
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Python path ayarlandı: {project_root}")
# --- PYTHONPATH Ayarı Sonu ---


# Gerekli modülleri içeri aktar
from src.config.base_config import BaseConfig
from src.config.model_config import ModelConfig
from src.model.components.layer_norm import LayerNorm
from src.model.components.embeddings import TokenEmbeddings, PositionalEmbeddings, EmbeddingLayer
from src.model.components.multi_head_attention import MultiHeadAttention
from src.model.components.feed_forward import FeedForward
from src.model.transformer_decoder_block import TransformerDecoderBlock
from src.model.echo_transformer import EchoTransformer
from src.utils.device_manager import DeviceManager
from src.utils.logger import setup_logging
from src.utils.checkpoint_manager import CheckpointManager


def run_all_component_tests():
    """
    Tamamlanmış model bileşenlerini ve konfigürasyonlarını test eder.
    """
    print("--- Tüm Tamamlanan Model Bileşenleri Testi Başlatılıyor ---")

    # 1. Konfigürasyon Dosyalarını Yükle ve Test Et
    print("\n1. Konfigürasyon Dosyaları Test Ediliyor...")
    try:
        base_cfg = BaseConfig()
        model_cfg = ModelConfig()
        print(f"  BaseConfig yüklendi: Cihaz={base_cfg.device}, Vocab Size={base_cfg.vocab_size}")
        print(f"  ModelConfig yüklendi: d_model={model_cfg.d_model}, n_layers={model_cfg.n_layers}")

        assert model_cfg.d_ff == model_cfg.d_model * 4, "ModelConfig d_ff otomatik ayarı başarısız!"
        print("  Konfigürasyon dosyaları başarıyla yüklendi ve varsayılan değerleri kontrol edildi.")
    except Exception as e:
        print(f"Konfigürasyon testi BAŞARISIZ: {e} ❌")
        return False
    print("Konfigürasyon Testi Tamamlandı. ✅")

    # 2. LayerNorm Katmanını Test Et
    print("\n2. LayerNorm Katmanı Test Ediliyor...")
    try:
        d_model_test = model_cfg.d_model
        eps_test = base_cfg.adam_epsilon 

        print("  --- RMS Normalizasyon Testi ---")
        rms_norm_layer = LayerNorm(d_model_test, eps=eps_test, bias=False, use_rms_norm=True)
        dummy_input = torch.randn(2, 10, d_model_test)
        output_rms_custom = rms_norm_layer(dummy_input)
        mean_of_squares_rms = output_rms_custom.pow(2).mean(-1).mean().item()
        assert abs(mean_of_squares_rms - 1.0) < 0.1, \
            f"  RMSNorm testi başarısız: ortalama kare {mean_of_squares_rms:.6f} (Beklenen ~1.0)"
        print("  RMSNorm testi başarılı.")

        print("  --- Standart Layer Normalizasyon Testi ---")
        std_norm_layer_custom = LayerNorm(d_model_test, eps=eps_test, bias=True, use_rms_norm=False)
        output_std_custom = std_norm_layer_custom(dummy_input)
        std_norm_layer_pytorch = nn.LayerNorm(d_model_test, eps=eps_test)
        std_norm_layer_pytorch.weight.data = std_norm_layer_custom.weight.data
        if std_norm_layer_custom.bias is not None:
            std_norm_layer_pytorch.bias.data = std_norm_layer_custom.bias.data
        output_std_pytorch = std_norm_layer_pytorch(dummy_input)
        tolerance = 1e-4
        assert torch.allclose(output_std_custom, output_std_pytorch, atol=tolerance), \
            "  Standart LayerNorm testi başarısız: Özel implementasyon PyTorch'a uymuyor!"
        print("  Standart LayerNorm testi başarılı!")

    except Exception as e:
        print(f"LayerNorm testi BAŞARISIZ: {e} ❌")
        return False
    print("LayerNorm Testi Tamamlandı. ✅")

    # 3. Embeddings Katmanlarını Test Et
    print("\n3. Embeddings Katmanları Test Ediliyor...")
    try:
        _vocab_size = base_cfg.vocab_size
        _d_model = model_cfg.d_model
        _max_seq_len = base_cfg.max_seq_len

        batch_size_embed = 2
        seq_len_embed = 128
        dummy_input_ids = torch.randint(0, _vocab_size, (batch_size_embed, seq_len_embed))

        token_embed_test = TokenEmbeddings(_vocab_size, _d_model)
        token_output = token_embed_test(dummy_input_ids)
        assert token_output.shape == (batch_size_embed, seq_len_embed, _d_model), "  TokenEmbeddings boyutu yanlış!"
        print("  TokenEmbeddings testi başarılı.")

        pos_embed_test = PositionalEmbeddings(_d_model, _max_seq_len)
        dummy_input_for_pos = torch.randn(batch_size_embed, seq_len_embed, _d_model)
        pos_output = pos_embed_test(dummy_input_for_pos)
        assert pos_output.shape == (batch_size_embed, seq_len_embed, _d_model), "  PositionalEmbeddings boyutu yanlış!"
        assert not torch.equal(dummy_input_for_pos, pos_output), "  Konumsal gömme eklenmedi!"
        print("  PositionalEmbeddings testi başarılı.")

        embedding_layer = EmbeddingLayer(_vocab_size, model_cfg, base_cfg)
        embedding_output = embedding_layer(dummy_input_ids)
        assert embedding_output.shape == (batch_size_embed, seq_len_embed, _d_model), "  EmbeddingLayer boyutu yanlış!"
        print("  EmbeddingLayer testi başarılı.")

    except Exception as e:
        print(f"Embeddings testi BAŞARISIZ: {e} ❌")
        return False
    print("Embeddings Testi Tamamlandı. ✅")

    # 4. MultiHeadAttention Katmanı Test Et
    print("\n4. MultiHeadAttention Katmanı Test Ediliyor...")
    try:
        batch_size_attn = 2
        seq_len_attn = 32
        d_model_attn = model_cfg.d_model
        n_heads_attn = model_cfg.n_heads

        dummy_input_attn = torch.randn(batch_size_attn, seq_len_attn, d_model_attn)
        print(f"  Giriş tensörü boyutu: {dummy_input_attn.shape}")

        attention_layer = MultiHeadAttention(model_config=model_cfg, base_config=base_cfg)

        # Test Senaryosu 1: Temel ileri besleme (maskesiz, cache'siz)
        print("  --- Test Senaryosu 1: Temel İleri Besleme (Maskesiz, Cache'siz) ---")
        output_s1, present_kv_s1 = attention_layer(dummy_input_attn)
        assert output_s1.shape == (batch_size_attn, seq_len_attn, d_model_attn), "  S1 çıktı boyutu yanlış!"
        assert present_kv_s1 is None, "  S1 Cache beklenmedi ama döndürüldü!"
        print("  Test Senaryosu 1 başarılı. ✅")

        # Test Senaryosu 2: Cache kullanımı (üretim modu)
        print("  --- Test Senaryosu 2: Cache Kullanımı (Metin Üretimi Modu) ---")
        single_token_input = torch.randn(batch_size_attn, 1, d_model_attn)
        output_step1, present_kv_step1 = attention_layer(single_token_input, use_cache=True)
        assert output_step1.shape == (batch_size_attn, 1, d_model_attn), "  Adım 1 çıktı boyutu yanlış!"
        assert present_kv_step1 is not None, "  Adım 1 Cache bekleniyordu ama döndürülmedi!"
        assert present_kv_step1[0].shape == (batch_size_attn, n_heads_attn, 1, d_model_attn // n_heads_attn), "  Adım 1 Cache K boyutu yanlış!"

        single_token_input_step2 = torch.randn(batch_size_attn, 1, d_model_attn)
        output_step2, present_kv_step2 = attention_layer(single_token_input_step2, layer_past=present_kv_step1, use_cache=True)
        assert output_step2.shape == (batch_size_attn, 1, d_model_attn), "  Adım 2 çıktı boyutu yanlış!"
        assert present_kv_step2 is not None, "  Adım 2 Cache bekleniyordu ama döndürülmedi!"
        assert present_kv_step2[0].shape == (batch_size_attn, n_heads_attn, 2, d_model_attn // n_heads_attn), "  Adım 2 Cache K boyutu yanlış!"
        print("  Test Senaryosu 2 başarılı. ✅")

        # Test Senaryosu 3: Padding Maskesi Kullanımı
        print("  --- Test Senaryosu 3: Padding Maskesi Kullanımı ---")
        padding_mask_input = torch.ones(batch_size_attn, seq_len_attn, dtype=torch.bool)
        padding_mask_input[0, seq_len_attn-2:] = False
        padding_mask_input[1, seq_len_attn-5:] = False

        output_with_mask, _ = attention_layer(dummy_input_attn, attention_mask=padding_mask_input)
        assert output_with_mask.shape == (batch_size_attn, seq_len_attn, d_model_attn), "  Maskeli çıktı boyutu yanlış!"
        print("  Test Senaryosu 3 başarılı. ✅")

    except Exception as e:
        print(f"MultiHeadAttention testi BAŞARISIZ: {e} ❌")
        return False
    print("MultiHeadAttention Testi Tamamlandı. ✅")

    # 5. FeedForward Katmanı Test Et
    print("\n5. FeedForward Katmanı Test Ediliyor...")
    try:
        batch_size_ff = 2
        seq_len_ff = 32
        d_model_ff = model_cfg.d_model
        d_ff_test = model_cfg.d_ff

        dummy_input_ff = torch.randn(batch_size_ff, seq_len_ff, d_model_ff)
        print(f"  Giriş tensörü boyutu: {dummy_input_ff.shape}")

        feed_forward_layer = FeedForward(model_config=model_cfg)

        output_ff = feed_forward_layer(dummy_input_ff)
        print(f"  Çıktı boyutu: {output_ff.shape} (Beklenen: {batch_size_ff, seq_len_ff, d_model_ff})")
        assert output_ff.shape == (batch_size_ff, seq_len_ff, d_model_ff), "  FeedForward çıktı boyutu yanlış!"

        print("  FeedForward katmanı testi başarılı. ✅")

    except Exception as e:
        print(f"FeedForward testi BAŞARISIZ: {e} ❌")
        return False
    print("FeedForward Testi Tamamlandı. ✅")
    
    # 6. TransformerDecoderBlock Katmanı Test Et
    print("\n6. TransformerDecoderBlock Katmanı Test Ediliyor...")
    try:
        batch_size_block = 2
        seq_len_block = 32
        d_model_block = model_cfg.d_model

        dummy_input_block = torch.randn(batch_size_block, seq_len_block, d_model_block)
        print(f"  Giriş tensörü boyutu: {dummy_input_block.shape}")

        decoder_block_layer = TransformerDecoderBlock(model_config=model_cfg, base_config=base_cfg)

        # Test Senaryosu 1: Temel ileri besleme (maskesiz, cache'siz)
        print("  --- Test Senaryosu 1: Temel İleri Besleme (Maskesiz, Cache'siz) ---")
        output_s1_block, present_kv_s1_block = decoder_block_layer(dummy_input_block)
        assert output_s1_block.shape == (batch_size_block, seq_len_block, d_model_block), "  Block S1 çıktı boyutu yanlış!"
        assert present_kv_s1_block is None, "  Block S1 Cache beklenmedi ama döndürüldü!"
        print("  Test Senaryosu 1 başarılı. ✅")

        # Test Senaryosu 2: Cache kullanımı (metin üretimi modu)
        print("  --- Test Senaryosu 2: Cache Kullanımı (Metin Üretimi Modu) ---")
        single_token_input_block = torch.randn(batch_size_block, 1, d_model_block)
        output_step1_block, present_kv_step1_block = decoder_block_layer(single_token_input_block, use_cache=True)
        assert output_step1_block.shape == (batch_size_block, 1, d_model_block), "  Block Adım 1 çıktı boyutu yanlış!"
        assert present_kv_step1_block is not None, "  Block Adım 1 Cache bekleniyordu ama döndürülmedi!"

        single_token_input_step2_block = torch.randn(batch_size_block, 1, d_model_block)
        output_step2_block, present_kv_step2_block = decoder_block_layer(single_token_input_step2_block, layer_past=present_kv_step1_block, use_cache=True)
        assert output_step2_block.shape == (batch_size_block, 1, d_model_block), "  Block Adım 2 çıktı boyutu yanlış!"
        assert present_kv_step2_block is not None, "  Adım 2 Cache bekleniyordu ama döndürülmedi!"
        assert present_kv_step2_block[0].shape == (batch_size_block, model_cfg.n_heads, 2, model_cfg.d_model // model_cfg.n_heads), "  Block Adım 2 Cache K boyutu yanlış!"
        print("  Test Senaryosu 2 başarılı. ✅")

        # Test Senaryosu 3: Padding Maskesi Kullanımı
        print("  --- Test Senaryosu 3: Padding Maskesi Kullanımı ---")
        padding_mask_block = torch.ones(batch_size_block, seq_len_block, dtype=torch.bool)
        padding_mask_block[0, seq_len_block-2:] = False
        padding_mask_block[1, seq_len_block-5:] = False

        output_with_mask_block, _ = decoder_block_layer(dummy_input_block, attention_mask=padding_mask_block)
        assert output_with_mask_block.shape == (batch_size_block, seq_len_block, d_model_block), "  Block Maskeli çıktı boyutu yanlış!"
        print("  Test Senaryosu 3 başarılı. ✅")

    except Exception as e:
        print(f"TransformerDecoderBlock testi BAŞARISIZ: {e} ❌")
        return False
    print("TransformerDecoderBlock Testi Tamamlandı. ✅")

    # 7. EchoTransformer (Ana Model) Katmanı Test Et
    print("\n7. EchoTransformer (Ana Model) Katmanı Test Ediliyor...")
    try:
        # Modeli oluştur
        model = EchoTransformer(base_config=base_cfg, model_config=model_cfg)
        print(f"  EchoTransformer modeli başarıyla oluşturuldu. Toplam parametre: {model.num_params() / 1e6:.2f} Milyon.")
        
        # Modelin doğru cihazda olup olmadığını kontrol et (eğer cuda varsa)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"  Model cihazda: {next(model.parameters()).device}")

        # Test girdileri
        batch_size_main = 2
        seq_len_main = 16 
        dummy_input_ids_main = torch.randint(0, base_cfg.vocab_size, (batch_size_main, seq_len_main)).to(device)
        dummy_attention_mask_main = torch.ones(batch_size_main, seq_len_main, dtype=torch.bool).to(device)
        
        # Test Senaryosu 1: Temel ileri besleme (cache'siz)
        print("  --- Test Senaryosu 1: Temel İleri Besleme (Cache'siz) ---")
        logits_s1_main, present_kv_s1_main = model(dummy_input_ids_main, attention_mask=dummy_attention_mask_main, use_cache=False)
        assert logits_s1_main.shape == (batch_size_main, seq_len_main, base_cfg.vocab_size), "  S1 logits boyutu yanlış!"
        assert present_kv_s1_main is None, "  S1 Cache beklenmedi ama döndürüldü!"
        print("  Test Senaryosu 1 başarılı. ✅")

        # Test Senaryosu 2: Cache kullanımı (metin üretimi modu)
        print("  --- Test Senaryosu 2: Cache Kullanımı (Metin Üretimi Modu) ---")
        # İlk token için forward
        first_token_input_ids = torch.randint(0, base_cfg.vocab_size, (batch_size_main, 1)).to(device)
        first_token_attention_mask = torch.ones(batch_size_main, 1, dtype=torch.bool).to(device)
        
        logits_step1, present_kv_step1 = model(first_token_input_ids, attention_mask=first_token_attention_mask, use_cache=True)
        assert logits_step1.shape == (batch_size_main, 1, base_cfg.vocab_size), "  Adım 1 logits boyutu yanlış!"
        assert present_kv_step1 is not None, "  Adım 1 Cache bekleniyordu ama döndürülmedi!"
        assert len(present_kv_step1) == model_cfg.n_layers, "  Adım 1 Cache listesi uzunluğu yanlış!"
        print("  Adım 1 başarılı. ✅")

        # İkinci token için forward (önceki cache ile)
        second_token_input_ids = torch.randint(0, base_cfg.vocab_size, (batch_size_main, 1)).to(device)
        second_token_attention_mask = torch.ones(batch_size_main, 2, dtype=torch.bool).to(device) # Total seq_len = 2

        logits_step2, present_kv_step2 = model(
            second_token_input_ids,
            attention_mask=second_token_attention_mask,
            past_key_values=present_kv_step1,
            use_cache=True
        )
        assert logits_step2.shape == (batch_size_main, 1, base_cfg.vocab_size), "  Adım 2 logits boyutu yanlış!"
        assert present_kv_step2 is not None, "  Adım 2 Cache bekleniyordu ama döndürülmedi!"
        assert len(present_kv_step2) == model_cfg.n_layers, "  Adım 2 Cache listesi uzunluğu yanlış!"
        assert present_kv_step2[0][0].shape[-2] == 2, "  Adım 2 Cache boyutu yanlış!"
        print("  Adım 2 başarılı. ✅")

    except Exception as e:
        print(f"EchoTransformer testi BAŞARISIZ: {e} ❌")
        return False
    print("EchoTransformer Testi Tamamlandı. ✅")

    # 8. DeviceManager Katmanı Test Et
    print("\n8. DeviceManager Katmanı Test Ediliyor...")
    try:
        # Otomatik cihaz seçimi testi
        print("  --- Otomatik Cihaz Seçimi ---")
        dm_auto = DeviceManager(device_name="auto")
        dummy_tensor_auto = torch.randn(2, 2)
        dummy_tensor_auto_on_device = dm_auto.to_device(dummy_tensor_auto)
        assert dummy_tensor_auto_on_device.device == dm_auto.current_device, "  Otomatik cihaz seçimi yanlış!"
        print(f"  Tensör cihazda: {dummy_tensor_auto_on_device.device}")
        dm_auto.print_device_info()

        # CPU seçimi testi
        print("\n  --- CPU Cihaz Seçimi ---")
        dm_cpu = DeviceManager(device_name="cpu")
        dummy_tensor_cpu = torch.randn(2, 2)
        dummy_tensor_cpu_on_device = dm_cpu.to_device(dummy_tensor_cpu)
        assert dummy_tensor_cpu_on_device.device.type == "cpu", "  CPU seçimi yanlış!"
        print(f"  Tensör cihazda: {dummy_tensor_cpu_on_device.device}")
        dm_cpu.print_device_info()

        # CUDA seçimi testi (eğer GPU varsa)
        if torch.cuda.is_available():
            print("\n  --- CUDA Cihaz Seçimi ---")
            dm_cuda = DeviceManager(device_name="cuda")
            dummy_tensor_cuda = torch.randn(2, 2)
            dummy_tensor_cuda_on_device = dm_cuda.to_device(dummy_tensor_cuda)
            assert dummy_tensor_cuda_on_device.device.type == "cuda", "  CUDA seçimi yanlış!"
            print(f"  Tensör cihazda: {dummy_tensor_cuda_on_device.device}")
            dm_cuda.print_device_info()
        else:
            print("\n  --- CUDA Cihaz Seçimi (GPU mevcut değil) ---")
            print("  CUDA cihaz mevcut değil, test atlandı veya CPU'ya düşüldü.")

        # Geçersiz cihaz adı testi
        print("\n  --- Geçersiz Cihaz Adı Testi ---")
        try:
            DeviceManager(device_name="invalid_device_name")
            assert False, "  Geçersiz cihaz adı hatası bekleniyordu ama alınmadı!"
        except ValueError as e:
            print(f"  Hata yakalandı (beklenen): {e}")
        
        print("  DeviceManager testleri başarılı. ✅")

    except Exception as e:
        print(f"DeviceManager testi BAŞARISIZ: {e} ❌")
        return False
    print("DeviceManager Testi Tamamlandı. ✅")

    # 9. Logger Test Ediliyor...
    # Test klasörlerini temizle (önceki çalıştırmalardan kalmış olabilir)
    test_output_dir_log = "test_runs_logger"
    # logging.getLogger().handlers listesindeki FileHandler'ları kapat
    # Bu, Windows'daki PermissionError'ı çözer
    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logging.root.removeHandler(handler)

    if os.path.exists(test_output_dir_log):
        shutil.rmtree(test_output_dir_log)
    try:
        # Loglama sistemini kur
        test_run_name_log = "logger_functional_test"
        setup_logging(log_level="DEBUG", output_dir=test_output_dir_log, run_name=test_run_name_log)
        
        # Bir logger al ve mesajlar gönder
        test_logger = logging.getLogger("test_logger")
        
        test_logger.debug("  Bu bir DEBUG mesajıdır.")
        test_logger.info("  Bu bir INFO mesajıdır.")
        test_logger.warning("  Bu bir WARNING mesajıdır.")
        test_logger.error("  Bu bir ERROR mesajıdır.")
        test_logger.critical("  Bu bir CRITICAL mesajıdır.")

        # Log dosyasının oluşturulup oluşturulmadığını ve boş olmadığını kontrol et
        log_dir_path = os.path.join(test_output_dir_log, test_run_name_log, "logs")
        log_files = [f for f in os.listdir(log_dir_path) if f.endswith('.log')]
        assert len(log_files) > 0, "  Log dosyası oluşturulmadı!"
        
        # İlk log dosyasının boyutunu kontrol et
        first_log_file_path = os.path.join(log_dir_path, log_files[0])
        assert os.path.getsize(first_log_file_path) > 0, "  Log dosyası boş!"
        
        # Konsol çıktısı manuel kontrol gerektirir, ancak fonksiyonun çalıştığını doğrulamış olduk.
        print(f"  Loglar '{log_dir_path}' dizininde oluşturuldu ve başarıyla yazıldı.")
        print("  Logger testi başarılı. ✅")

    except Exception as e:
        print(f"Logger testi BAŞARISIZ: {e} ❌")
        return False
    finally:
        # Oluşturulan test dizinlerini temizle
        # Bu temizlik artık PermissionError'a yol açmamalıdır.
        if os.path.exists("test_runs_dm"):
            shutil.rmtree("test_runs_dm")
        if os.path.exists("test_runs_ckpt"):
            shutil.rmtree("test_runs_ckpt")
        if os.path.exists(test_output_dir_log):
            shutil.rmtree(test_output_dir_log)
        if os.path.exists("test_runs"):
            shutil.rmtree("test_runs")


    print("\n--- Tüm Tamamlanan Model Bileşenleri Testleri Başarıyla Tamamlandı! ---")
    return True

if __name__ == "__main__":
    if run_all_component_tests():
        print("\nTüm testler başarıyla geçti. Devam edebiliriz! 🎉")
    else:
        print("\nBazı testler başarısız oldu. Lütfen hataları kontrol edin. ❌")