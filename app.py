import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from datetime import datetime, timedelta
import os

import numpy as np
import jModel

def complete_model_predict(input1, input2, bill_jModel, event_jModel, final_jModel):
    model1 = bill_jModel.predict(input1).reshape(1, 545, 1)
    model2 = event_jModel.predict(input2)
    model3 = np.concatenate([model1, model2], axis=2)
    return final_jModel.predict(model3)

class PredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("Sales Recapitulation Analysis")

        self.model = None
        self.model_path = 'model/model.keras'
        self.model_loaded = False

        self.file_paths = [None, None]

        self.label = tk.Label(master, text="Upload CSV files:")
        self.label.pack()

        self.csv1_frame = tk.Frame(master)
        self.csv1_frame.pack(fill=tk.BOTH, expand=True)

        self.csv1_table = ttk.Treeview(self.csv1_frame)
        self.csv1_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.csv1_scrollbar = ttk.Scrollbar(self.csv1_frame, orient="vertical", command=self.csv1_table.yview)
        self.csv1_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.csv1_table.configure(yscrollcommand=self.csv1_scrollbar.set)

        self.csv2_frame = tk.Frame(master)
        self.csv2_frame.pack(fill=tk.BOTH, expand=True)

        self.csv2_table = ttk.Treeview(self.csv2_frame)
        self.csv2_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.csv2_scrollbar = ttk.Scrollbar(self.csv2_frame, orient="vertical", command=self.csv2_table.yview)
        self.csv2_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.csv2_table.configure(yscrollcommand=self.csv2_scrollbar.set)

        self.button1 = tk.Button(master, text="Upload CSV1", command=lambda: self.upload_csv(0))
        self.button1.pack()

        self.button2 = tk.Button(master, text="Upload CSV2", command=lambda: self.upload_csv(1))
        self.button2.pack()

        self.load_model_button = tk.Button(master, text="Load Model", command=self.load_model)
        self.load_model_button.pack()

        self.predict_button = tk.Button(master, text="Predict", command=self.predict)
        self.predict_button.pack()

    def upload_csv(self, index):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.file_paths[index] = file_path
                print(f"CSV{index + 1} uploaded: {file_path}")

                if index == 0:
                    self.display_csv_data(file_path, self.csv1_table, index)
                elif index == 1:
                    self.display_csv_data(file_path, self.csv2_table, index)
        except Exception as e:
            messagebox.showerror("Alert", f"Error when upload data!, {str(e)}")

    def display_csv_data(self, file_path, table, index):
        try:
            data = pd.read_csv(file_path)
            if len(data) != 180 and index == 0:
                messagebox.showerror("Alert", "Data 1 should have length 180 days!")
                return
            if len(data) != 545 and index == 1:
                messagebox.showerror("Alert", "Data 2 should have length 545 days!")
                return
            table.delete(*table.get_children())
            table['column'] = list(data.columns)
            table['show'] = 'headings'
            
            # Set column widths based on content width
            max_widths = [max([len(str(row[i])) for row in data.values] + [len(str(data.columns[i]))]) for i in range(len(data.columns))]
            for i, column in enumerate(table['columns']):
                table.heading(column, text=column)
                table.column(column, width=max_widths[i]*10)  # Adjust the factor to set the width based on content length
            
            for row in data.values:
                table.insert('', 'end', values=row)
            
            messagebox.showinfo("Success", "Data loaded successfully!")
        except Exception as e:
            messagebox.showerror("Alert", f"Error when read data!, {str(e)}")
            print(f"Error displaying CSV data: {str(e)}")


    def load_model(self):
        global bill_model, event_model, final_model
        try:
            bill_model = jModel.Model()
            bill_model.load_model('model/bill_model.jmodel')

            event_model = jModel.Model()
            event_model.load_model('model/event_model.jmodel')

            final_model = jModel.Model()
            final_model.load_model('model/final_model.jmodel')
            self.model_loaded = True
            print("Model loaded successfully!")
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Alert", f"Error when loading model!, {str(e)}")
            print(f"Error loading model: {str(e)}")
            self.model_loaded = False

    def predict(self):
        if not self.model_loaded:
            messagebox.showerror("Alert", "Model not loaded!")
            print("Model not loaded!")
            return

        if None in self.file_paths:
            messagebox.showerror("Alert", "Please upload both CSV files.")
            print("Please upload both CSV files.")
            return

        try:
            df_data1 = pd.read_csv(self.file_paths[0])
            data1 = df_data1.values[:, 1:].reshape(1, 180, 1).astype(float)
            data1 = (data1 - 7)/77
            data2 = pd.read_csv(self.file_paths[1]).values[:, 1:].reshape(1, 545, 12).astype(float)

            start_date = datetime.strptime(df_data1.iloc[-1, 0], '%d/%m/%Y')
            end_date = start_date + timedelta(days=365)
            date_range = pd.date_range(start=start_date+timedelta(days=1), end=end_date)
            df_predict = pd.DataFrame(date_range, columns=['Order Date'])

            predictions = complete_model_predict(data1, data2, bill_model, event_model, final_model)
            predictions = 7 + predictions*77
            predictions = predictions.astype(int)

            df_predict['Bill Count'] = predictions
            folder_name = 'result'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                print(f"Folder '{folder_name}' created successfully.")
            else:
                print(f"Folder '{folder_name}' already exists.")
            df_predict.to_csv("result/predictions.csv", index=False)
            messagebox.showinfo("Success", "Prediction success!, the result is in 'result/predictions.csv'")
            print("Predictions saved to predictions.csv")
        except Exception as e:
            messagebox.showerror("Alert", f"Error when making prediction!, {str(e)}")
            print(f"Error making predictions: {str(e)}")

def main():
    root = tk.Tk()
    root.title("Sales Recapitulation Analysis")
    root.geometry("800x600")  # Set the fixed size of the window
    root.resizable(False, False)
    app = PredictionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
