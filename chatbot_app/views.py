from django.shortcuts import render, HttpResponse
import tensorflow as tf
import json
import numpy as np
from django.http import JsonResponse


# Create your views here.
def index(request):
    return render(request, 'index.html')

def contact(request):
    return HttpResponse("contact us")





def process_input(request):
    user_input = request.POST.get('user_input')

    with open("chatbot_app/chatbotmodel/tokenizer_word_index.json", "r") as f:
        word_index = json.load(f)

    interpreter = tf.lite.Interpreter(model_path="chatbot_app/chatbotmodel/chatbot_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def predict(input_text):
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.word_index = word_index

        seed_text = input_text
        next_word = " "
        answer = ""
        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        # Accuracy of right question
        question_accuracy = len(token_list) / len(input_text.split(" "))
        question_accuracy = int(question_accuracy * 100)

        if question_accuracy < 50:
            low_acc_query = f"Pardon! I can't process this query. QA: {question_accuracy}%, Is it a valid question?"
            return low_acc_query

        while next_word != "ouavjra":
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=input_details[0]['shape'][1], padding='pre')

            input_data = token_list.astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])

            predicted = np.argmax(output_data, axis=1)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
            next_word = output_word

        try:
            answer = seed_text.split("arjvauo")[1][:-7]
        except:
            answer = "Pardon! I didn't find an answer."

        return answer

    response = predict(user_input)

    return JsonResponse({'response': response})

