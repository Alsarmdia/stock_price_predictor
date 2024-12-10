import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class StockPricePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Starbucks Stock Price Predictor")
        self.root.geometry("900x600")

        # Initialize attributes
        self.data = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.fig = None  # To store the current figure for export

        # Track dark mode state
        self.dark_mode = False

        # Create the GUI components
        self.create_widgets()

    def create_widgets(self):
        # Frame for Buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Buttons
        self.add_button("Load Dataset", self.load_data, "Load a CSV file containing stock price data.")
        self.add_button("Visualize Closing Prices", self.plot_closing_prices, "Visualize the closing prices over time.")
        self.add_button("Train Model", self.train_model, "Train a linear regression model on the dataset.")
        self.add_button("Show Predictions", self.show_predictions, "Show actual vs predicted closing prices.")
        self.add_button("Show Correlation Heatmap", self.plot_correlation, "Show correlation heatmap of features.")
        self.add_button("Toggle Dark Mode", self.toggle_dark_mode, "Switch between dark mode and light mode.")
        self.add_button("Export Graph", self.export_graph, "Export the current graph as a PNG file.")
        self.add_button("Quit", self.root.quit, "Exit the application.", fg="red")

        # Frame for Graphs
        self.graph_frame = tk.Frame(self.root)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas to display graphs
        self.canvas = None

    def add_button(self, text, command, tooltip_text, fg="black"):
        btn = tk.Button(self.button_frame, text=text, command=command, width=25, fg=fg)
        btn.pack(pady=10)
        self.create_tooltip(btn, tooltip_text)

    def create_tooltip(self, widget, text):
        tooltip = tk.Toplevel(widget, bg="white")
        tooltip.wm_overrideredirect(True)
        tooltip.withdraw()

        label = tk.Label(tooltip, text=text, bg="#333", fg="#fff", relief=tk.SOLID, borderwidth=1, font=("Arial", 9))
        label.pack()

        def show_tooltip(event):
            tooltip.deiconify()
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 50
            y += widget.winfo_rooty() + 20
            tooltip.wm_geometry(f"+{x}+{y}")

        def hide_tooltip(event):
            tooltip.withdraw()

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

    def toggle_dark_mode(self):
        if self.dark_mode:
            self.root.config(bg="SystemButtonFace")
            self.button_frame.config(bg="SystemButtonFace")
            self.graph_frame.config(bg="SystemButtonFace")
            fg_color = "black"
            bg_color = "SystemButtonFace"
        else:
            self.root.config(bg="#2E2E2E")
            self.button_frame.config(bg="#2E2E2E")
            self.graph_frame.config(bg="#2E2E2E")
            fg_color = "white"
            bg_color = "#2E2E2E"

        for widget in self.button_frame.winfo_children():
            widget.config(fg=fg_color, bg=bg_color)

        self.dark_mode = not self.dark_mode

    def load_data(self):
        file_path = filedialog.askopenfilename(title="Select a CSV File", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            messagebox.showinfo("Success", "Dataset loaded successfully!")
        else:
            messagebox.showerror("Error", "No file selected.")

    def clear_graph_frame(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.fig = None  # Clear the stored figure

    def plot_closing_prices(self):
        if self.data is not None:
            self.clear_graph_frame()

            self.fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(self.data['datetime'], self.data['close'], label='Closing Price', color='blue')
            ax.set_title('Starbucks Closing Stock Prices Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Closing Price (USD)')
            ax.legend()
            ax.grid(True)

            self.display_graph(self.fig)
        else:
            messagebox.showerror("Error", "Please load the dataset first.")

    def train_model(self):
        if self.data is not None:
            self.data['price_change'] = self.data['close'] - self.data['open']
            self.data['high_low_range'] = self.data['high'] - self.data['low']
            self.data['sma_7'] = self.data['close'].rolling(window=7).mean()
            self.data['sma_30'] = self.data['close'].rolling(window=30).mean()

            features = ['open', 'high', 'low', 'volume', 'price_change', 'high_low_range', 'sma_7', 'sma_30']
            self.data = self.data.dropna()
            X = self.data[features]
            y = self.data['close']

            X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

            self.model = LinearRegression()
            self.model.fit(X_train, y_train)

            messagebox.showinfo("Success", "Model trained successfully!")
        else:
            messagebox.showerror("Error", "Please load the dataset first.")

    def show_predictions(self):
        if self.model is not None and self.X_test is not None:
            self.clear_graph_frame()

            predictions = self.model.predict(self.X_test)

            self.fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(self.data['datetime'].iloc[-len(self.y_test):], self.y_test, label='Actual', color='green')
            ax.plot(self.data['datetime'].iloc[-len(self.y_test):], predictions, label='Predicted', color='red')
            ax.set_title('Actual vs Predicted Closing Prices')
            ax.set_xlabel('Date')
            ax.set_ylabel('Closing Price (USD)')
            ax.legend()
            ax.grid(True)

            self.display_graph(self.fig)
        else:
            messagebox.showerror("Error", "Please train the model first.")

    def plot_correlation(self):
        if self.data is not None:
            self.clear_graph_frame()

            self.fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(self.data[['open', 'high', 'low', 'close', 'volume']].corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Matrix of Stock Prices and Volume')

            self.display_graph(self.fig)
        else:
            messagebox.showerror("Error", "Please load the dataset first.")

    def display_graph(self, fig):
        self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def export_graph(self):
        if self.fig:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                self.fig.savefig(file_path)
                messagebox.showinfo("Success", f"Graph saved to {file_path}")
        else:
            messagebox.showerror("Error", "No graph to export.")

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPricePredictorApp(root)
    root.mainloop()