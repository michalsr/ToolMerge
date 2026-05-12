"""Answerer prompt used by Qwen3-VL 8f paper rows."""

ANSWERER_V1 = """\
Based on the video frames shown, answer the following question.

Question: {question}
Options:
{options}

Select the best answer and respond with the letter ({option_letters})."""
