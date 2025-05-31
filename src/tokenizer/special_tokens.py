
# src/tokenizer/special_tokens.py

# Bu modül, dil modeli için kullanılacak özel token'ları tanımlar.
# Bu token'lar genellikle modelin ve tokenleyicinin davranışını yönlendirmek için kullanılır.

# Başlangıç ve Bitiş Tokenları (Beginning-of-Sentence, End-of-Sentence)
# Metin üretimi ve sıralı girdi işleme için kullanılır.
BOS_TOKEN: str = "<|endoftext|>" # Genellikle GPT-2'de EOS olarak kullanılır, bizde BOS olarak alalım
EOS_TOKEN: str = "<|endoftext|>" # Genellikle metinlerin sonunu belirtir

# Padding Token (Doldurma Tokenı)
# Farklı uzunluktaki dizileri aynı batçede işleyebilmek için kısa dizilerin sonuna eklenir.
PAD_TOKEN: str = "<|pad|>"

# Bilinmeyen Token (Out-of-Vocabulary)
# Tokenleyici tarafından bilinmeyen kelimeler veya karakterler için kullanılır.
UNK_TOKEN: str = "<|unk|>"

# Tokenleyici ID'leri (Hugging Face tarzı)
# Tokenleyiciden sonra bu string'lerin karşılık geldiği ID'ler de tutulabilir.
# Ancak başlangıçta sadece string olarak tanımlıyoruz.
# Tokenizer eğitildikten sonra bu ID'ler belirlenecektir.
# Varsayılan ID'ler için genellikle HF'nin gpt-2 tokenizer'ı baz alınır.
# (OLMO'da da gpt-neox tokenizer kullanılır ve bazı özel token ID'leri vardır.)

# Örneğin, GPT-2'nin özel token ID'leri:
# EOS_TOKEN_ID = 50256 (aynı zamanda BOS gibi kullanılabilir)
# PAD_TOKEN_ID = 50257 (eklenmiş olabilir)
# UNK_TOKEN_ID = 50257 (eğer vocab boyutu 50257 ise)

# Bu ID'ler, tokenleyici eğitildikten sonra güncellenecektir.
# Şimdilik, konfigürasyonda varsayılan vocab_size'ı (50257) kullanıyoruz.
# Tokenizer eğitiminden sonra bu ID'ler belirlenecek ve BaseConfig'e veya ayrı bir config'e eklenebilir.

# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("Special Tokens testi başlatılıyor...")

    print(f"BOS Token: {BOS_TOKEN}")
    print(f"EOS Token: {EOS_TOKEN}")
    print(f"PAD Token: {PAD_TOKEN}")
    print(f"UNK Token: {UNK_TOKEN}")

    assert isinstance(BOS_TOKEN, str), "BOS_TOKEN string değil!"
    assert len(BOS_TOKEN) > 0, "BOS_TOKEN boş!"
    print("Special Tokens tanımları başarılı. ✅")