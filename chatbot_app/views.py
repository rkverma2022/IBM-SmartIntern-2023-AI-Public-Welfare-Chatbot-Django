from django.shortcuts import render, HttpResponse
import tensorflow as tf
import json
import numpy as np
from django.http import JsonResponse
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def WQ(message):
    try:
        data = pd.read_csv("chatbot_app\water_dataset.csv")
        data = data.drop(['id'], axis=1)
        d = data
        d = d.drop(['Potability'], axis=1)
        maxValueForNormalization = d.max().to_numpy(dtype=float)

        data = data / data.max()

        data = data.to_numpy()
        x = data[:, :-1]
        y = data[:, -1]

        # X = message

        # X = np.array([X], dtype=float) / maxValueForNormalization
        X = message / maxValueForNormalization
        X = X.reshape(1, 9)

        KNN = KNeighborsClassifier()
        KNN.fit(x, y)

        result = KNN.predict(X)

        if result == 0:
            return "Water Parameters Shows that it is not safe for drinking"
        elif result == 1:
            return "Water is safe ! You can drink it "
    except FileNotFoundError:
        return "File not found! Please check the file path."
    except (ValueError, IndexError):
        return "Data format issue or missing parameters."

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
    
    def checkwq(message):
        global wqAnswer
        helpingSentence = ""
        try:
            mess = message
            # mess = mess.split(" ")[1]
            mess = mess.split(',')
            mess = np.array(mess, dtype=float)
            wqAnswer = ""
            if mess.shape[0] == 9:
                helpingSentence = ""

                wqAnswer = WQ(mess)

            elif mess.shape[0] < 9:
                helpingSentence = "Some Parameters are Missing"

            else:
                helpingSentence = "Remove unnecessary extra values"

            # print(f"{mess}  Type {type(mess)} dtype {mess.dtype} {mess.shape}")
        except:
            wqAnswer = "Sorry I can't Process This"
            helpingSentence = "Thank You"
        
        return wqAnswer + helpingSentence

   

    if "predict water quality" in user_input or "predict quality of water" in user_input:
        try:
            user_input = str(user_input).split("{")[1].split("}")[0]
            print(user_input)
            response = checkwq(user_input)
        except:
            response = '''
    I usually process the following 9 parameters
    
    
    1. ph: pH of 1. water (0 to 14).
    2. Hardness: Capacity of water to precipitate soap in mg/L.
    3. Solids: Total dissolved solids in ppm.
    4. Chloramines: Amount of Chloramines in ppm.
    5. Sulfate: Amount of Sulfates dissolved in mg/L.
    6. Conductivity: Electrical conductivity of water in μS/cm.
    7. Organic_carbon: Amount of organic carbon in ppm.
    8. Trihalomethanes: Amount of Trihalomethanes in μg/L.
    9. Turbidity: Measure of light emiting property of water in NTU.
    

    Instructions : 
    
    1. Please Don't put any space, use comma to separate the parameters values
    2. Please Provide all 9 parameters
    
    Ex: 
    9,145,13168,9,310,592,8,77.2,3.8
    
    
    Thank You !'''
    else:
        response = predict(user_input)

    return JsonResponse({'response': response})

