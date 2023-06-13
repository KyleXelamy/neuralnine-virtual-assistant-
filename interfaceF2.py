import tkinter as tk
import speech_recognition as sr
import pyttsx3
import matplotlib.pyplot as plt

# Initialize the speech recognition and text-to-speech engines
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Create the Tkinter window
window = tk.Tk()
window.title("Voice Assistant")

# Create a text box to display the conversation
conversation_text = tk.Text(window, height=10, width=50)
conversation_text.pack()

# Create a figure for the visualizer
figure = plt.figure(figsize=(5, 4))
visualizer = figure.add_subplot(111)
visualizer.set_axis_off()
visualizer.imshow(plt.imread("microphone_icon.png"))

def speak(text):
    engine.say(text)
    engine.runAndWait()

def process_input():
    user_input = user_input_entry.get()
    conversation_text.insert(tk.END, "User: " + user_input + "\n")
    user_input_entry.delete(0, tk.END)

    intents_list = predict_intent(user_input)
    response = get_response(intents_list, intents)
    conversation_text.insert(tk.END, "Assistant: " + response + "\n")
    speak(response)

def process_voice_input():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        user_input = recognizer.recognize_google(audio)
        conversation_text.insert(tk.END, "User: " + user_input + "\n")

        intents_list = predict_intent(user_input)
        response = get_response(intents_list, intents)
        conversation_text.insert(tk.END, "Assistant: " + response + "\n")
        speak(response)
    except sr.UnknownValueError:
        conversation_text.insert(tk.END, "Assistant: Sorry, I didn't understand that. Could you please repeat?\n")
        speak("Sorry, I didn't understand that. Could you please repeat?")
    except sr.RequestError:
        conversation_text.insert(tk.END, "Assistant: Sorry, I'm having trouble processing your request. Please try again later.\n")
        speak("Sorry, I'm having trouble processing your request. Please try again later.")

# Create a button to process text input
text_input_button = tk.Button(window, text="Send", command=process_input)
text_input_button.pack()

# Create a microphone icon button to process voice input
def microphone_click():
    visualizer.clear()
    visualizer.imshow(plt.imread("microphone_icon_active.png"))
    window.update()
    process_voice_input()
    visualizer.clear()
    visualizer.imshow(plt.imread("microphone_icon.png"))
    window.update()

microphone_icon = plt.imshow(plt.imread("microphone_icon.png"))
microphone_button = tk.Button(window, image=microphone_icon, command=microphone_click, bd=0)
microphone_button.pack()

# Set up the visualizer in the window
visualizer_canvas = FigureCanvasTkAgg(figure, master=window)
visualizer_canvas.draw()
visualizer_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Start the Tkinter event loop
window.mainloop()
