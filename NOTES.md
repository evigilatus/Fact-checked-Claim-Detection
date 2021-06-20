# Fact-checked-Claim-Detection

## Ресурси
### CLEF 2020, Task 2:
  - [Описание на задача](https://docs.google.com/document/d/1yNSdiwK_0EOuabtN1XXvqMfmyp406VtmzpH5FLwR5_M/edit#)
  - [Github repo](https://github.com/sshaar/clef2020-factchecking-task2)
  - [Статия на Преслав Наков](https://arxiv.org/pdf/2005.06058.pdf?fbclid=IwAR3Ta6vILOYZAr8AdMkHlR7m_5kgvsIJC0-X2RwPfKvrGxsiEruX0m99SSc)
  - [Статия на победителите](http://ceur-ws.org/Vol-2696/paper_134.pdf)
  - [Overview на използваните подходи в състезанието](https://arxiv.org/pdf/2007.07997v1.pdf)


## Бележки

### Поуки от победителите:
- Размерите на допълнителните датасети трябва да съвпадат с размера на оригиналния.
- Ако огментираме данните, етикетите трябва да са до частите от оригиналния запис, който описват.
- Добре е да се генерират допълнителни данни (превод?)
- Тренирането не трябва да е върху всички документи, защото е небалансирано (да се направи груба подборка).
- Тренировачните данни да са балансирани между положителни и отрицателни примери.


### Линкове от Момчил Back Translation
- [Back Translation](https://www.aclweb.org/anthology/D18-1045.pdf)
- [Hugging-Face Translator](https://huggingface.co/transformers/model_doc/marian.html?highlight=opus)
- [Open NTM Translator](https://opennmt.net/)
- [FaceBook FairSeq Translator](https://github.com/pytorch/fairseq)

### Експерименти
- [Описание и коментари](https://docs.google.com/document/d/15G5nBYrVV3UATuqH6XN42YpSlAmC80JSpgOrjZvkVJQ/edit#heading=h.h651lqk1uo8r)
