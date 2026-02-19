import pdfplumber
import json
import re
import uuid

INPUT_PDF = "knowlage_base_NetGenius.pdf"
OUTPUT_JSON = "chunks.json"

TARGET_CHUNKS = 70
OVERLAP_PERCENT = 0.1  # 10% overlap


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text


def extract_pdf_content(pdf_path):
    pages_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text()
                if raw_text:
                    pages_content.append({
                        "page": page_num,
                        "text": clean_text(raw_text)
                    })
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return pages_content


def main():
    pages = extract_pdf_content(INPUT_PDF)

    if not pages:
        print("No text extracted.")
        return

    # Combine text with page tracking
    full_text = ""
    char_to_page = []

    for p in pages:
        start_idx = len(full_text)
        full_text += p["text"] + " "
        end_idx = len(full_text)
        char_to_page.append((start_idx, end_idx, p["page"]))

    total_chars = len(full_text)

    ideal_chunk_size = int(total_chars / TARGET_CHUNKS)
    overlap_size = int(ideal_chunk_size * OVERLAP_PERCENT)

    # Split by sentences (works with Arabic + English)
    sentences = re.split(r'(?<=[.!؟!])\s+', full_text)

    chunks = []
    current_sentences = []
    current_length = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        sent_len = len(sent)

        if current_length + sent_len > ideal_chunk_size and current_sentences:
            chunk_text = " ".join(current_sentences)

            # Determine page
            chunk_start_idx = full_text.find(current_sentences[0])
            page_num = 1
            for start, end, p_num in char_to_page:
                if start <= chunk_start_idx < end:
                    page_num = p_num
                    break

            chunks.append({
                "id": str(uuid.uuid4()),
                "page": page_num,
                "text": chunk_text
            })

            # Overlap logic
            overlap_sentences = []
            overlap_len = 0

            for s in reversed(current_sentences):
                if overlap_len + len(s) < overlap_size:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break

            current_sentences = overlap_sentences + [sent]
            current_length = sum(len(s) for s in current_sentences)

        else:
            current_sentences.append(sent)
            current_length += sent_len + 1

    # Add last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunk_start_idx = full_text.find(current_sentences[0])
        page_num = 1
        for start, end, p_num in char_to_page:
            if start <= chunk_start_idx < end:
                page_num = p_num
                break

        chunks.append({
            "id": str(uuid.uuid4()),
            "page": page_num,
            "text": chunk_text
        })

    print(f"Final total chunks: {len(chunks)}")

    # Save as proper JSON (array)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUTPUT_JSON}")
    print("==== Sample Preview ====")
    print(chunks[0]["text"])
    print("========================")


if __name__ == "__main__":
    main()
