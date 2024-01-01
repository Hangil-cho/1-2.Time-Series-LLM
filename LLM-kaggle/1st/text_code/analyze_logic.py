#텍스트 분석 기술에 대한 분서게 대해 찾아본 내용 (로직 관련)

'''######################################################################################################
1) 문장 길이 분석
 - 실제 에세이의 경우 길이를 정해놓고 하는 경우가 많아 AI로 만든 에세이보다 길이가 일정
######################################################################################################'''
def analyze_essay(essay):
    tokens = word_tokenize(essay)
    avg_sentence_length = sum(len(sentence.split()) for sentence in essay.split('.')) / len(essay.split('.'))
    return avg_sentence_length

'''######################################################################################################
2) 어휘의 다양성 분석
 - 실제 에세이의 경우 같은 말을 반복할 때 어색함을 느끼는 경우가 많아 다양한 어휘를 사용하지만
   AI의 경우, 같은 말을 반복하는데 거리낌이 없음 => 제한된 어휘 사용 혹은 사용빈도 불규칙적
######################################################################################################'''
def analyze_essay(essay):
    tokens = word_tokenize(essay)
    word_count = len(set(tokens))
    return word_count

'''######################################################################################################
3) 문법적 오류 분석
 - 실제 에세이의 경우 수차례 검토를 거치기 떄문에, 문법적 오류 발생
   하지만, AI의 경우 작성자의 의도 혹은 일반 문장에 대한 문법 학습이 완전하지 않다면 문법적 오류가 왕왕 발생할 수 있음
######################################################################################################'''
def analyze_essay(essay):
    tokens = word_tokenize(essay)
    tagged_tokens = pos_tag(tokens)
    grammar_errors = 0
    for token, tag in tagged_tokens:
        if tag == 'NN' and token[-1] == 's':
            grammar_errors += 1
    return grammar_errors

'''######################################################################################################
4) 언어 감정 분석
 - 실제 에세이의 경우 사람이 적기 때문에, 요약 혹은 결론, 과정 등에 감정이 섞인 어휘를 선택할 수 있지만,
   AI의 경우 그러한 어휘 선택없이 일괄적으로 몇개만 선택하여 사용함
######################################################################################################'''
def analyze_essay(essay):
    tokens = word_tokenize(essay)
    sentiment = TextBlob(essay).sentiment.polarity
    return sentiment



'''
요약본을 평가하기 위한 모델(품질)에 대한 컨테스트 https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview/efficiency-prize-evaluation
'''
