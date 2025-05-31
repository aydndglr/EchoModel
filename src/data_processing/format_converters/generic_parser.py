# src/data_processing/format_converters/generic_parser.py

import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Generator, Dict, Any
from pathlib import Path
import bz2
import gzip # Gzip sıkıştırmalı dosyalar için

log = logging.getLogger(__name__)

def parse_text_file(filepath: Path) -> Generator[Dict[str, str], None, None]:
    """
    Düz metin (.txt) dosyalarını okur ve her satırı veya paragrafı bir doküman olarak döndürür.
    Sıkıştırılmış dosyaları (gz, bz2) da otomatik olarak açar.

    Args:
        filepath (Path): Metin dosyasının yolu.

    Yields:
        Dict[str, str]: "text" anahtarıyla metin içeriğini içeren sözlük.
    """
    log.info(f"Düz metin dosyası '{filepath}' ayrıştırılıyor...")
    
    _open = open
    if filepath.suffix == '.gz':
        _open = gzip.open
    elif filepath.suffix == '.bz2':
        _open = bz2.open

    try:
        with _open(filepath, 'rt', encoding='utf-8') as f: # 'rt' metin modunda açar
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    yield {"text": stripped_line}
        log.info(f"Düz metin dosyası '{filepath}' ayrıştırma tamamlandı.")
    except Exception as e:
        log.error(f"'{filepath}' düz metin dosyası ayrıştırılırken hata: {e}", exc_info=True)


def parse_json_file(filepath: Path, text_field: str = "text") -> Generator[Dict[str, str], None, None]:
    """
    JSON veya JSONL dosyalarını okur ve belirtilen metin alanını çıkarır.
    Sıkıştırılmış dosyaları (gz, bz2) da destekler.

    Args:
        filepath (Path): JSON veya JSONL dosyasının yolu.
        text_field (str): Metin içeriğini içeren JSON anahtarı.

    Yields:
        Dict[str, str]: "text" anahtarıyla metin içeriğini içeren sözlük.
    """
    log.info(f"JSON/JSONL dosyası '{filepath}' ayrıştırılıyor...")

    _open = open
    if filepath.suffix == '.gz':
        _open = gzip.open
    elif filepath.suffix == '.bz2':
        _open = bz2.open

    try:
        # Dosyanın tek bir JSON objesi mi yoksa JSONL mi olduğunu anlamaya çalış
        # Büyük dosyalar için tümünü yüklemeden ilk birkaç satırı kontrol etmek daha iyi
        is_jsonl = True
        with _open(filepath, 'rt', encoding='utf-8') as f_peek:
            first_non_empty_line = ""
            for i, line in enumerate(f_peek):
                stripped_line = line.strip()
                if stripped_line:
                    first_non_empty_line = stripped_line
                    break
                if i > 100: # Çok fazla boş satır olmasın
                    break
            
            if first_non_empty_line.startswith('[') and first_non_empty_line.endswith(']'):
                # Başlangıç ve bitişte köşeli parantez varsa tek büyük JSON array'i
                # Ancak bu durumda tüm dosyayı yüklemek gerekecek, dikkat.
                # Büyük dosyalar için JSONL tercih edilir.
                # Şimdilik bunu bir JSONL olarak kabul etmeyelim.
                is_jsonl = False
            elif first_non_empty_line.startswith('{') and first_non_empty_line.endswith('}'):
                # Tek bir JSON objesi olabilir, ancak JSONL'de de tek bir JSON objesi olabilir satırda.
                # Bu çok belirsiz, ama varsayılan olarak JSONL'i tercih edelim
                # Çünkü LLM verileri genellikle büyük ve JSONL'dir.
                try:
                    json.loads(first_non_empty_line)
                    is_jsonl = True # İlk satır geçerli JSON ise JSONL varsayalım
                except json.JSONDecodeError:
                    is_jsonl = False # Bu durumda tek bir büyük JSON objesi veya bozuk dosya
            else:
                is_jsonl = True # Başka bir şeyse JSONL varsayalım

        if not is_jsonl:
            # Tüm dosyayı tek bir JSON objesi olarak yükle (küçük dosyalar için)
            log.info(f"'{filepath}' tek bir JSON objesi olarak ayrıştırılıyor...")
            with _open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and text_field in item and item[text_field] is not None:
                            text_content = str(item[text_field]).strip()
                            if text_content:
                                yield {"text": text_content}
                elif isinstance(data, dict) and text_field in data and data[text_field] is not None:
                    text_content = str(data[text_field]).strip()
                    if text_content:
                        yield {"text": text_content}
                else:
                    log.warning(f"JSON dosyası desteklenmeyen formatta: {type(data)} - {filepath}")
        else: # JSONL dosyası veya satır bazlı JSON
            log.info(f"'{filepath}' JSONL formatında ayrıştırılıyor...")
            with _open(filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        if text_field in data and data[text_field] is not None:
                            text_content = str(data[text_field]).strip()
                            if text_content:
                                yield {"text": text_content}
                    except json.JSONDecodeError as e:
                        log.warning(f"JSONL ayrıştırma hatası: {e} - Satır atlanıyor: {line.strip()[:100]}...")
    except Exception as e:
        log.error(f"'{filepath}' JSON/JSONL dosyası ayrıştırılırken hata: {e}", exc_info=True)
    
    log.info(f"JSON/JSONL dosyası '{filepath}' ayrıştırma tamamlandı.")


def parse_xml_file(filepath: Path, text_tag: str = "text") -> Generator[Dict[str, str], None, None]:
    """
    XML dosyalarını okur ve belirtilen etiket altındaki metin içeriğini çıkarır.
    Özellikle Wikipedia dump'ları gibi XML tabanlı veriler için kullanılır.
    Sıkıştırılmış dosyaları (gz, bz2) da destekler.

    Args:
        filepath (Path): XML dosyasının yolu.
        text_tag (str): Metin içeriğini içeren XML etiketinin adı (örn. "text" veya "{namespace_uri}text").

    Yields:
        Dict[str, str]: "text" anahtarıyla metin içeriğini içeren sözlük.
    """
    log.info(f"XML dosyası '{filepath}' ayrıştırılıyor...")
    
    _open = open
    if filepath.suffix == '.gz':
        _open = gzip.open
        # ET.iterparse gzip'i doğrudan desteklemeyebilir, o yüzden bz2 gibi açık hale getirmemiz lazım.
        # Python 3.8+ için ElementTree gzip/bz2 sıkıştırmalı dosya nesnelerini doğrudan kabul eder.
    elif filepath.suffix == '.bz2':
        _open = bz2.open

    try:
        # Namespace'i ele almak için. OLMO da bu şekilde yapıyor olabilir.
        # Örneğin: "{http://www.mediawiki.org/xml/export-0.10/}text"
        target_tag = text_tag
        if '{' not in text_tag and '}' not in text_tag:
            # Eğer tag'de namespace belirtilmemişse, iterparse'ı tüm namespace'ler için kullanırız
            # ve tag'in sadece yerel adını kontrol ederiz.
            pass # target_tag aynı kalır
        
        # Et.iterparse, dosya nesnesini bekler.
        with _open(filepath, 'rb') as f: # 'rb' binary modda açar (XML için)
            context = ET.iterparse(f, events=("end",)) # 'end' eventi elemanın bitişini yakalar

            for event, elem in context:
                if event == "end":
                    # Namespace'i olan tag'ler için elem.tag formatı '{uri}local_name' olur.
                    # Eğer text_tag'de namespace varsa tam eşleşme, yoksa sadece yerel adını kontrol et.
                    if (target_tag == elem.tag) or \
                       ('{' not in target_tag and elem.tag.endswith(target_tag)):
                        text_content = elem.text
                        if text_content:
                            yield {"text": text_content.strip()}
                        elem.clear() # Belleği serbest bırak (büyük XML'ler için kritik)
        log.info(f"XML dosyası '{filepath}' ayrıştırma tamamlandı.")

    except Exception as e:
        log.error(f"'{filepath}' XML dosyası ayrıştırılırken hata: {e}", exc_info=True)


# --- Test Fonksiyonu ---
if __name__ == "__main__":
    print("GenericParser testi başlatılıyor...")

    import shutil
    from src.utils.logger import setup_logging
    
    test_output_dir = "test_runs_generic_parser"
    test_run_name = "generic_parser_test_run"
    
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True) 

    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logging.root.removeHandler(handler)
    setup_logging(log_level="INFO", output_dir=test_output_dir, run_name=test_run_name)
    log = logging.getLogger(__name__)

    # Test Senaryosu 1: Düz Metin Dosyası (.txt)
    print("\n--- Test Senaryosu 1: Düz Metin Dosyası (.txt) ---")
    txt_file_path = Path(test_output_dir) / "test_doc.txt"
    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write("İlk paragraf.\n\nİkinci paragraf.\nÜçüncü paragraf.")
    texts_txt = list(parse_text_file(txt_file_path))
    assert len(texts_txt) == 3, f"TXT ayrıştırma hatası! Beklenen 3, bulunan {len(texts_txt)}"
    assert texts_txt[0]["text"] == "İlk paragraf.", "TXT ayrıştırma içeriği yanlış!"
    print("TXT ayrıştırma testi başarılı. ✅")
    
    # Test Senaryosu 1.1: GZ Sıkıştırmalı Metin Dosyası (.txt.gz)
    print("\n--- Test Senaryosu 1.1: GZ Sıkıştırmalı Metin Dosyası (.txt.gz) ---")
    gz_txt_file_path = Path(test_output_dir) / "test_doc.txt.gz"
    with gzip.open(gz_txt_file_path, 'wt', encoding='utf-8') as f:
        f.write("Sıkıştırılmış ilk paragraf.\nSıkıştırılmış ikinci paragraf.")
    texts_gz_txt = list(parse_text_file(gz_txt_file_path))
    assert len(texts_gz_txt) == 2, f"GZ TXT ayrıştırma hatası! Beklenen 2, bulunan {len(texts_gz_txt)}"
    assert texts_gz_txt[0]["text"] == "Sıkıştırılmış ilk paragraf.", "GZ TXT ayrıştırma içeriği yanlış!"
    print("GZ Sıkıştırmalı TXT ayrıştırma testi başarılı. ✅")

    # Test Senaryosu 2: JSONL Dosyası (.jsonl)
    print("\n--- Test Senaryosu 2: JSONL Dosyası (.jsonl) ---")
    jsonl_file_path = Path(test_output_dir) / "test_data.jsonl"
    with open(jsonl_file_path, "w", encoding="utf-8") as f:
        json.dump({"id": 1, "content": "JSONL ilk metin."}, f)
        f.write("\n")
        json.dump({"id": 2, "content": "JSONL ikinci metin."}, f)
        f.write("\n")
        json.dump({"id": 3, "content": None}, f)
        f.write("\n")
        f.write("Geçersiz JSON\n")
    texts_jsonl = list(parse_json_file(jsonl_file_path, text_field="content"))
    assert len(texts_jsonl) == 2, f"JSONL ayrıştırma hatası! Beklenen 2, bulunan {len(texts_jsonl)}"
    assert texts_jsonl[0]["text"] == "JSONL ilk metin.", "JSONL ayrıştırma içeriği yanlış!"
    print("JSONL ayrıştırma testi başarılı. ✅")
    
    # Test Senaryosu 2.1: Tek JSON Dosyası (.json)
    print("\n--- Test Senaryosu 2.1: Tek JSON Dosyası (.json) ---")
    single_json_path = Path(test_output_dir) / "single_data.json"
    with open(single_json_path, "w", encoding="utf-8") as f:
        json.dump({"title": "Tekil Başlık", "article": "Bu tek bir JSON dosyasından gelen makale metnidir."}, f)
    texts_single_json = list(parse_json_file(single_json_path, text_field="article"))
    assert len(texts_single_json) == 1, f"Tek JSON ayrıştırma hatası! Beklenen 1, bulunan {len(texts_single_json)}"
    assert texts_single_json[0]["text"] == "Bu tek bir JSON dosyasından gelen makale metnidir.", "Tek JSON içeriği yanlış!"
    print("Tek JSON ayrıştırma testi başarılı. ✅")

    # Test Senaryosu 3: XML Dosyası (.xml)
    print("\n--- Test Senaryosu 3: XML Dosyası (.xml) ---")
    xml_file_path = Path(test_output_dir) / "test_wiki.xml"
    with open(xml_file_path, "w", encoding="utf-8") as f:
        f.write("""
        <root>
            <page>
                <title>Sayfa 1</title>
                <text>Bu XML dosyasındaki ilk metin içeriğidir.</text>
            </page>
            <page>
                <title>Sayfa 2</title>
                <text>İkinci metin içeriği burada.</text>
            </page>
            <page>
                <title>Sayfa 3</title>
                <text></text>
            </page>
        </root>
        """)
    texts_xml = list(parse_xml_file(xml_file_path, text_tag="text"))
    assert len(texts_xml) == 2, f"XML ayrıştırma hatası! Beklenen 2, bulunan {len(texts_xml)}"
    assert texts_xml[0]["text"] == "Bu XML dosyasındaki ilk metin içeriğidir.", "XML ayrıştırma içeriği yanlış!"
    print("XML ayrıştırma testi başarılı. ✅")
    
    # Test Senaryosu 4: BZ2 Sıkıştırmalı XML Dosyası (.xml.bz2) (Wikipedia gibi)
    print("\n--- Test Senaryosu 4: BZ2 Sıkıştırmalı XML Dosyası (.xml.bz2) ---")
    bz2_xml_file_path = Path(test_output_dir) / "test_wiki.xml.bz2"
    xml_content_bytes = """
        <root>
            <page>
                <title>Sıkıştırılmış Sayfa 1</title>
                <text>Bu sıkıştırılmış XML dosyasındaki metin içeriğidir.</text>
            </page>
        </root>
    """.encode('utf-8')
    with bz2.open(bz2_xml_file_path, 'wb') as f:
        f.write(xml_content_bytes)
    texts_bz2_xml = list(parse_xml_file(bz2_xml_file_path, text_tag="text"))
    assert len(texts_bz2_xml) == 1, f"BZ2 XML ayrıştırma hatası! Beklenen 1, bulunan {len(texts_bz2_xml)}"
    assert texts_bz2_xml[0]["text"] == "Bu sıkıştırılmış XML dosyasındaki metin içeriğidir.", "BZ2 XML ayrıştırma içeriği yanlış!"
    print("BZ2 Sıkıştırmalı XML ayrıştırma testi başarılı. ✅")


    print("\nGenericParser tüm testleri tamamlandı. ✅")

    # Genel temizlik
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
        log.info(f"Temizlik yapıldı: '{test_output_dir}' silindi.")