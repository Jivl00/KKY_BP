import tkinter as tk
from tkinter import ttk
import intent_classifier_keras
import intent_classifier_pytorch

net = intent_classifier_pytorch

# Load model
model = net.load_best_model()

# Root window
window = tk.Tk()
window.title('Intent Classifier')

window_width = 512
window_height = 128

# Get the screen dimension
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Find the center point
center_x = int(screen_width / 2 - window_width / 2)
center_y = int(screen_height / 2 - window_height / 2)

# Set the position of the window to the center of the screen
window.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
window.iconbitmap('ikonka.ico')

# Frame
frame = ttk.Frame(window)

# Field options
options = {'padx': 5, 'pady': 5}

# Enter label
enter = ttk.Label(frame, text='Enter sentence:', font=("calibri", 15, "bold"))
enter.grid(column=0, row=0, rowspan=2, sticky='W', **options)

# Sentence entry
sentence = tk.StringVar()
sentence_entry = ttk.Entry(frame, textvariable=sentence)
sentence_entry.grid(column=1, row=0, rowspan=2, columnspan=3, **options)
sentence_entry.focus()


def classify_button_clicked():
    """  Handle classify button click event
    """
    global model
    input_sentence = sentence.get()
    model = net.load_best_model()
    pred_class = net.predict_single(input_sentence, model)

    result = 'Predicted class: {}'.format(pred_class.upper())
    result_label.config(text=result, font=("calibri", 15))


def enter(event):
    """  Pressing enter on keyboard does the same as pressing classify button
    """
    if event.keysym == 'Return':
        classify_button_clicked()


window.bind('<Return>', enter)

# Create style Object
style = ttk.Style()

style.configure('TButton', font=('calibri', 15, 'bold'), borderwidth='4')

# Classify button
classify_button = ttk.Button(frame, text='Classify')
classify_button.grid(column=4, row=0, rowspan=2, sticky='W', **options)
classify_button.configure(command=classify_button_clicked)

# Result label
result_label = ttk.Label(frame)
result_label.grid(row=3, columnspan=3, **options)


def print_selection():
    global net
    if var.get() == 0:
        net = intent_classifier_pytorch
    else:
        net = intent_classifier_keras


var = tk.IntVar()

c1 = tk.Radiobutton(frame, text='Pytorch', variable=var, value=0, command=print_selection, anchor='w')
c1.grid(row=0, column=6, rowspan=1, columnspan=3, sticky='SW')

c2 = tk.Radiobutton(frame, text='Keras', variable=var, value=1, command=print_selection, anchor='w')
c2.grid(row=1, column=6, rowspan=1, columnspan=3, sticky='NW')

# Add padding to the frame and show it
frame.grid(padx=20, pady=20)

# Start the app
window.mainloop()
