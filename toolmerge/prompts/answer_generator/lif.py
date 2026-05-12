"""Answerer prompt used by every paper run (Tables 2-5)."""

ANSWERER_LIF = """\
Select the best answer to the following multiple-choice question based on the video.
Question: {question}
Options: {options}
Answer with the option's letter from the given choices directly.
Your response format should be strictly an upper case letter {option_letters}."""
