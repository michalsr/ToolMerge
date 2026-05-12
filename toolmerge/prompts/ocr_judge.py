"""OCR judge prompts.

The LLM judge decides whether each OCR-extracted text snippet helps answer
the question. Frames whose text gets a YES verdict are inserted at rank 1
during merging.

Two variants:
    ``OCR_JUDGE_PROMPT``        — one snippet per call.
    ``OCR_JUDGE_BATCH_PROMPT``  — many snippets in one call (set
                                   ``ocr_batch_size > 1`` to enable).
"""

OCR_JUDGE_PROMPT = """\
Does this on-screen text help determine which answer choice is correct?

Important: if the question quotes a subtitle or on-screen text to describe WHEN something happens (e.g. "when the subtitle says X, what..."), that quoted text is usually just a timestamp cue. However, if the on-screen text is a word-for-word match (or very close match) to the quoted subtitle in the question, answer YES — that frame marks the exact moment the question is asking about.

Question: {question}
Choices:
{options}

Text: "{ocr_text}"

Answer YES or NO."""


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
