"""OCR judge prompt.

The LLM judge decides whether OCR-extracted text snippets from video frames
help answer the question. Used at cache-build time only — see
``cache_build/ocr_judge.py``. Frames whose text gets a YES verdict are
inserted at rank 1 during inference-time merging.
"""

OCR_JUDGE_BATCH_PROMPT = """\
You are judging whether on-screen text from a video helps determine which answer choice is correct.

Important: if the question quotes a subtitle or on-screen text to describe WHEN something happens (e.g. "when the subtitle says X, what..."), that quoted text is usually just a timestamp cue. However, if the on-screen text is a word-for-word match (or very close match) to the quoted subtitle in the question, mark YES — that frame marks the exact moment the question is asking about. Otherwise mark NO.

Question: {question}
Choices:
{options}

Below are {n_texts} text snippets extracted from video frames via OCR. For EACH snippet, decide whether it helps determine the correct answer.

Reply with EXACTLY {n_texts} lines, one per snippet, in the format:
<number>. YES or NO

{text_list}"""
