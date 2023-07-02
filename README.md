# Task-2_mercor
# SHER_AI_Assistant
_@owner and @author: **Rajkumar Verma**_

_**SHER AI Assistant is a tool of SHER APP to help users to interact.**___

_**Instruction to clone and run the repository**_

      1. clone the repository, 


            git clone https://github.com/rkverma2022/Task-2_mercor.git

      2. install the required module

          pip install -r requirementss.txt

      3. To run

          python manage.py runserver

      4. _Keep In mind: I have trained the model with very less data (Nearly-40 questions)._

                  NLP DNN Model.zip file is containig all the code and file to generate the chatbot_model.tflite and tokenizer_word_index.json file

            Model Performance:
                  Epoch 29/30
                  29/29 [==============================] - 6s 213ms/step - loss: 0.1154 - accuracy: 0.9481
                  Epoch 30/30
                  29/29 [==============================] - 6s 213ms/step - loss: 0.1160 - accuracy: 0.9503
                              
          
            

## Overview of SHER_AI_Assistant

                    
_**Preparing the Data in json in Question and Answer formate.**_


                        |
                        |

                        
**Putting All the Questions and Answer in a single string formate**


                   i.e.

                   
                   corpus = 
                    '''Hello ARJVAUO hello, i am A I chat bot. How may i help you . OUAVJRA
                    Good morning ARJVAUO Good morning to you too! I'm here to assist you with any questions or concerns you may have. OUAVJRA
                    Good Afternoon ARJVAUO Good afternoon to you too! I'm here to assist you with any questions or concerns you may have. OUAVJRA
                    Good evening ARJVAUO Good evening to you too! I'm here to assist you with any questions or concerns you may have. OUAVJRA'''



                          |
                          | 

                          
**Tokenize using tokenizer And make ngram then Pad it**
        
        
        i.e.

        
        corpus = data.lower().split("\n")

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(corpus)
        self.total_words = len(self.tokenizer.word_index) + 1

        input_sequences = []
        for line in corpus:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)

        self.max_sequence_len = max(len(x) for x in input_sequences)
        input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=self.max_sequence_len, padding='pre')

        
                      |
                      |

                      
**Input (Xs) and Output (Ys)**


        i.e.
  
        
          xs = input_sequences[:, :-1]
          labels = input_sequences[:, -1]
          ys = tf.keras.utils.to_categorical(labels, num_classes=self.total_words)

        
                      |
                      |

                      
**Define the DNN (Dense Neural Network)**


        i.e.


        
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Embedding(self.total_words, 240, input_length=self.max_sequence_len - 1))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)))
        self.model.add(tf.keras.layers.Dense(self.total_words, activation='softmax'))
        adam = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        
                      |
                      |

                      
**Train the model**


        i.e.      

        
        history = self.model.fit(xs, ys, epochs=30, verbose=1)

        
                      |
                      |

                      
**Convert to model to chatbot_model.tflite and tokenized corpus to tokenizer_word_index.json**


        i.e.

        
        # Save the model as a SavedModel
        tf.saved_model.save(self.model, "chatbot_model")

        # Convert the model to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_saved_model("chatbot_model")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()


        # Save the TensorFlow Lite model to a file
        with open("chatbot_model.tflite", "wb") as f:
            f.write(tflite_model)

        # Save the tokenizer word index
        with open("tokenizer_word_index.json", "w") as f:
            json.dump(self.tokenizer.word_index, f)

            
                      |
                      |

                      
**Ready to Deploy to Django**


**Made a django Project and pass the both the tflite model and tokenizer_word_index.json file to chatbotmodel directory, at the app level**


                      |
                      |

                      
**Made a simple UI using html and CSS and define a form to take input and place to add messeage/response**


        i.e.
        
        
          <div class="chatbot-container">
            <div class="chat-header">
              <h3>SHER  AI  Assistant</h3>
            </div>
            <div class="chat-body" id="chat-body">
              <!-- Messages will be dynamically added here -->
            </div>
            <form class="chat-input" onsubmit="sendMessage(); return false;">
              {% csrf_token %}
              <input type="text" id="user-input" placeholder="Type your message..." />
              <input type="submit" value="Send" />
            </form>
          </div>

        
                                      |
                                      |

                                      
**used java script to take the input and  post it on pressing submit button**


       i.e.
    
       
          function sendMessage() {
          var userInput = document.getElementById("user-input");
          var userMessage = userInput.value;
          displayMessageWithDelay(userMessage, "user-message", 0); // Display user message instantly
          userInput.value = "";
    
          fetch('/process_input/', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/x-www-form-urlencoded',
              'X-CSRFToken': getCookie('csrftoken')
            },
            body: 'user_input=' + encodeURIComponent(userMessage)
          })
            .then(response => response.json())
            .then(data => {
              var botMessage = data.response;
              displayMessageWithDelay(botMessage, "bot-message", 50); // Display bot message with a delay of 50 milliseconds per character
            })
            .catch(error => {
              console.error('Error:', error);
            });
        }
    
        
                              |
                              |

                          
**In the views.py, call the process_input() function and pass the request to invoke the tokenizer_word_index.json and chatbot_model.tflite**
                         
                          
                          |
                          |

                          
**tokenize the input string posted from the user using  tokenizer_word_index.json**


**passed that tokenized string to the model to predict output and finally return the json response**


        i.e.
      
      
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
    
      
                              
                              |
                              |

                          
**Display the output (calling displayoutput javascript function from index.html)**


        i.e.
        
        
        function displayMessageWithDelay(message, sender, delay) {
              var chatBody = document.getElementById("chat-body");
              var messageDiv = document.createElement("div");
              messageDiv.className = "message " + sender;
              chatBody.appendChild(messageDiv);
        
              var index = 0;
        
              var displayNextCharacter = function () {
                if (index < message.length) {
                  messageDiv.textContent += message.charAt(index);
                  index++;
                  setTimeout(displayNextCharacter, delay);
                }
              };
        
              displayNextCharacter();
              chatBody.scrollTop = chatBody.scrollHeight;
            }
        
                                      |
                                      |

                              
_**Finally Output Has been displayed**__
                  
  

                                    
