import os
import re
import csv
import pypdf
from typing import List, Dict, Any

def chunk_text(text: str, target_words: int = 300, max_words: int = 450) -> List[str]:
    """Splits a long section into paragraph-aware chunks of target_words to max_words."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        words = para.split()
        word_count = len(words)
        
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_word_count = word_count
        else:
            current_chunk.append(para)
            current_word_count += word_count

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

def parse_handbook(file_path: str = "data/handbook.md") -> List[Dict[str, Any]]:
    chunks = []
    if not os.path.exists(file_path):
        return chunks

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    section_pattern = re.compile(r'^(#{1,3}\s+.*)$', re.MULTILINE)
    parts = section_pattern.split(content)

    current_section = "University Academic Handbook"
    file_name = "handbook.md"
    chunk_index = 0

    for i in range(0, len(parts)):
        part = parts[i].strip()
        if not part:
            continue
        
        if part.startswith("#"):
            header_title = re.sub(r'^#{1,3}\s+', '', part).strip()
            current_section = header_title
        else:
            text_passages = chunk_text(part)
            for passage in text_passages:
                chunk_index += 1
                chunk_id = f"handbook_{chunk_index:03d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "file": file_name,
                    "section": current_section,
                    "page": None,
                    "content": f"Section: {current_section}\n\n{passage}"
                })

    return chunks

def parse_academic_regulations(file_path: str = "data/academic_regulations.pdf") -> List[Dict[str, Any]]:
    chunks = []
    if not os.path.exists(file_path):
        return chunks

    reader = pypdf.PdfReader(file_path)
    file_name = "academic_regulations.pdf"
    chunk_index = 0

    section_regex = re.compile(r'^(\d+(?:\.\d+)*\s+[A-Z].*)$', re.MULTILINE)

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_text = page.extract_text() or ""
        
        cleaned_lines = []
        for line in page_text.splitlines():
            sline = line.strip()
            if sline in ["Academic Regulations", f"Page {page_num}", "University Academic Regulations", "Office of the Registrar - Synthetic Benchmark Edition"]:
                continue
            cleaned_lines.append(line)
        
        cleaned_text = "\n".join(cleaned_lines).strip()
        if not cleaned_text:
            continue

        headings = list(section_regex.finditer(cleaned_text))
        if headings:
            last_end = 0
            curr_section = "Academic Regulations"
            for h in headings:
                section_heading = h.group(1).strip()
                prev_text = cleaned_text[last_end:h.start()].strip()
                if prev_text:
                    passages = chunk_text(prev_text)
                    for passage in passages:
                        chunk_index += 1
                        chunks.append({
                            "chunk_id": f"academic_regulations_{chunk_index:03d}",
                            "file": file_name,
                            "section": curr_section,
                            "page": page_num,
                            "content": f"Section: {curr_section} (Page {page_num})\n\n{passage}"
                        })
                curr_section = section_heading
                last_end = h.end()
            
            remaining_text = cleaned_text[last_end:].strip()
            if remaining_text:
                passages = chunk_text(remaining_text)
                for passage in passages:
                    chunk_index += 1
                    chunks.append({
                        "chunk_id": f"academic_regulations_{chunk_index:03d}",
                        "file": file_name,
                        "section": curr_section,
                        "page": page_num,
                        "content": f"Section: {curr_section} (Page {page_num})\n\n{passage}"
                    })
        else:
            curr_section = "Academic Regulations"
            passages = chunk_text(cleaned_text)
            for passage in passages:
                chunk_index += 1
                chunks.append({
                    "chunk_id": f"academic_regulations_{chunk_index:03d}",
                    "file": file_name,
                    "section": curr_section,
                    "page": page_num,
                    "content": f"Page {page_num}\n\n{passage}"
                })

    return chunks

def parse_fee_schedule(file_path: str = "data/fee_schedule.csv") -> List[Dict[str, Any]]:
    chunks = []
    if not os.path.exists(file_path):
        return chunks

    file_name = "fee_schedule.csv"
    chunk_index = 0

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunk_index += 1
            cat = row.get("fee_category", f"Fee Item {chunk_index}")
            deadline = row.get("payment_deadline", "N/A")
            penalties = row.get("late_payment_penalties", "N/A")
            refunds = row.get("pro_rata_tuition_refund_percentages", "N/A")
            
            passage = (
                f"Fee Category: {cat}\n"
                f"Payment Deadline: {deadline}\n"
                f"Late Payment Penalties: {penalties}\n"
                f"Pro-Rata Tuition Refund Percentages: {refunds}"
            )
            
            chunks.append({
                "chunk_id": f"fee_schedule_{chunk_index:03d}",
                "file": file_name,
                "section": cat,
                "page": None,
                "content": passage
            })

    return chunks

def parse_custom_file(file_path: str, file_name: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        return []
    
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".md" or ext == ".txt":
        return parse_handbook(file_path)
    elif ext == ".pdf":
        return parse_academic_regulations(file_path)
    elif ext == ".csv":
        return parse_fee_schedule(file_path)
    else:
        # Generic text chunker fallback
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        passages = chunk_text(content)
        chunks = []
        for idx, p in enumerate(passages, 1):
            chunks.append({
                "chunk_id": f"{os.path.basename(file_name).replace('.', '_')}_{idx:03d}",
                "file": file_name,
                "section": "General Document",
                "page": None,
                "content": p
            })
        return chunks

def parse_all_documents(data_dir: str = "data") -> List[Dict[str, Any]]:
    all_chunks = []
    if not os.path.exists(data_dir):
        return all_chunks
    
    for fname in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext == ".md":
            all_chunks.extend(parse_handbook(fpath))
        elif ext == ".pdf":
            all_chunks.extend(parse_academic_regulations(fpath))
        elif ext == ".csv":
            all_chunks.extend(parse_fee_schedule(fpath))
        elif ext in [".txt"]:
            all_chunks.extend(parse_custom_file(fpath, fname))
            
    return all_chunks

if __name__ == "__main__":
    chunks = parse_all_documents()
    print(f"Total chunks parsed: {len(chunks)}")
    if chunks:
        print("Sample chunk:", chunks[0])

