import re


def replace_digits(text: str):
    text = re.sub('0', 'O', text)
    text = re.sub('1', 'I', text)
    text = re.sub('2', 'Z', text)
    text = re.sub('4', 'A', text)
    text = re.sub('5', 'S', text)
    text = re.sub('8', 'B', text)
    return text


def replace_letters(text: str):
    text = re.sub('O', '0', text)
    text = re.sub('Q', '0', text)
    text = re.sub('U', '0', text)
    text = re.sub('D', '0', text)
    text = re.sub('I', '1', text)
    text = re.sub('Z', '2', text)
    text = re.sub('A', '4', text)
    text = re.sub('S', '5', text)
    text = re.sub('B', '8', text)
    return text


def replace_sex(text: str):
    text = re.sub('P', 'F', text)
    text = re.sub('N', 'M', text)
    return text
