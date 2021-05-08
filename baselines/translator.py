# pylint:disable=missing-docstring
"""
from transformers import MarianMTModel, MarianTokenizer

src_text = [
    u">>fra<< this is a sentence in english that we want to translate to french",
    u">>por<< This should go to portuguese",
    u">>esp<< And this to Spanish",
]

model_name = "Helsinki-NLP/opus-mt-en-roa"
tokenizer = MarianTokenizer.from_pretrained(model_name)
print(tokenizer.supported_language_codes)
# ['>>zlm_Latn<<', '>>mfe<<', '>>hat<<', '>>pap<<',
# '>>ast<<', '>>cat<<', '>>ind<<', '>>glg<<', '>>wln<<', '>>spa<<', '>>fra<<',
# '>>ron<<', '>>por<<', '>>ita<<', '>>oci<<', '>>arg<<', '>>min<<']

model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
print([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
# ["c'est une phrase en anglais que nous voulons traduire en français",
# 'Isto deve ir para o português.',
# 'Y esto al español']
"""

import csv
import os
import json
from glob import glob
from nltk.tokenize import sent_tokenize
from transformers import MarianMTModel, MarianTokenizer


class Translator:
    EN_AR = ("Helsinki-NLP/opus-mt-en-ar", ">>ara<<")
    AR_EN = ("Helsinki-NLP/opus-mt-ar-en", ">>eng<<")

    def __init__(self, model_name, lang_prefix):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.lang_prefix = lang_prefix

    def translate(self, src_text):
        translated = self.model.generate(
            **self.tokenizer(
                self.prepare_text(src_text), return_tensors="pt", padding=True
            )
        )
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    def prepare_text(self, text):
        if isinstance(text, str):
            return self.lang_prefix + " " + text
        else:
            return [self.lang_prefix + " " + line for line in text]

    @staticmethod
    def back_translate(forward_model, backward_model, text):
        tgt_text = forward_model.translate(sent_tokenize(text))
        return " ".join(backward_model.translate(tgt_text))


def back_translate_tsv_tweets():
    dir = os.path.join("..", "data", "subtask-2a--english")
    file_name = "processed-tweets-train-dev.tsv"
    out_file = file_name.replace(".tsv", ".tr.tsv")

    with open(
        os.path.join(dir, file_name), newline="", encoding="utf-8"
    ) as ifile, open(
        os.path.join(dir, out_file), "w", newline="", encoding="utf-8"
    ) as ofile:
        reader = csv.reader(ifile, delimiter="\t")
        writer = csv.writer(ofile, delimiter="\t")
        for tweet_id, text in reader:
            arab = en_ar.translate(sent_tokenize(text))
            eng = ar_en.translate(arab)
            writer.writerow([tweet_id, " ".join(eng)])


def back_translate_tweet_jsons():
    dir = os.path.join("..", "data", "subtask-2a--english", "vclaims")
    file_names = glob(dir + "/*.json")
    for file_name in sorted(file_names):
        out_name = file_name.replace(".json", "_tr.json")
        with open(file_name) as input, open(out_name, "w") as output:
            content = json.load(input)
            if content.get("title"):
                content["title"] = Translator.back_translate(
                    en_ar, ar_en, content["title"]
                )
            if content.get("subtitle"):
                content["subtitle"] = Translator.back_translate(
                    en_ar, ar_en, content["subtitle"]
                )
            if content.get("vclaim"):
                content["vclaim"] = Translator.back_translate(
                    en_ar, ar_en, content["vclaim"]
                )
            json.dump(content, output, indent=None)


def translate_processed_tweets(en_ar, ar_en):
    dir = os.path.join(os.path.dirname(__file__), "..", "data", "subtask-2a--english", "test_data")
    print(dir)
    input_file_name = os.path.join(dir, "processed-tweets-test.tsv")
    output_file_name = input_file_name.replace(".tsv", "_tr.tsv")
    lines = []
    with open(input_file_name, newline="", encoding="utf-8") as input, open(
        output_file_name, "w", newline="", encoding="utf-8"
    ) as output:
        reader = csv.reader(input, delimiter="\t")
        writer = csv.writer(output, delimiter="\t")
        for tweet_id, text in reader:
            new_text = Translator.back_translate(en_ar, ar_en, text)
            lines.append((tweet_id, new_text))
        writer.writerows(lines)


if __name__ == "__main__":
    en_ar = Translator(*Translator.EN_AR)
    ar_en = Translator(*Translator.AR_EN)
    dir = os.path.join(os.path.dirname(__file__), "..", "data", "subtask-2b--english", "politifact-vclaims")
    print(dir)
    file_names = glob(dir + "/*.json")
    for file_name in sorted(file_names):
        out_file_name = file_name.replace("politifact", "translated")
        with open(file_name) as input, open(out_file_name, "w") as output:
            print(out_file_name)
            content = json.load(input)
            if content.get("title"):
                content["title"] = Translator.back_translate(
                    en_ar, ar_en, content["title"]
                )
            if content.get("text"):
                content["text"] = Translator.back_translate(
                    en_ar, ar_en, content["text"]
                )
            if content.get("vclaim"):
                content["vclaim"] = Translator.back_translate(
                    en_ar, ar_en, content["vclaim"]
                )
            json.dump(content, output, indent=None)
