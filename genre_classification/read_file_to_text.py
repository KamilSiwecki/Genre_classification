import os
from typing import List


def read_file_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    text = text.replace("\n", " ")
    cleaned_text = "".join([char for char in text if char.isalpha() or char == " "])
    cf_removed_text = " ".join([word for word in cleaned_text.split() if word != "cf"])
    return cf_removed_text


def read_directory(directory_path: str) -> List[str]:
    texts_list = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt') or file_name.endswith('.rtf'):
            file_text = read_file_text(file_path=os.path.join(directory_path, file_name))
            texts_list.append(file_text)
    return texts_list
