Here's your updated and **cleaned-up** `README.md` file with `results/` removed from all image paths so that the images load properly if the markdown is rendered from the `results/` folder or the root (depending on how you link them or host):

---

```markdown
# 🚗 CO₂ Emissions Regression App

Predict CO₂ emissions (g/km) from car specifications using machine learning models such as Polynomial and Spline Regression. This project demonstrates a complete ML pipeline: from EDA and model training to deployment using Streamlit.

---

## 📌 Project Overview

With increasing climate concerns and environmental regulations, vehicle emissions are a key metric. Using a Canadian vehicle emissions dataset, this project aims to:

- Analyze how engine specs and fuel consumption relate to CO₂ emissions
- Compare Polynomial and Spline regression models
- Build an interactive web app to predict emissions
- Visualize key trends for better understanding and decision-making

---

## 👨‍💻 Author

> Developed by **Tamaghna Nag**  
> 📍 London, UK | Kolkata, India  
> 🌐 [https://tamaghnatech.in](https://tamaghnatech.in)  
> 📧 tamaghnanag04@gmail.com  
> 🔗 [LinkedIn](https://www.linkedin.com/in/tamaghna99/) | [GitHub](https://github.com/Tamaghnatech)

---

## 🧠 Dataset Features

The original dataset contains information about:

- Engine Size (L)
- Number of Cylinders
- Fuel Consumption (City, Hwy, Comb)
- Fuel Type (D, E, X, Z, N)
- CO₂ Emissions (g/km)
- Fuel Consumption (mpg)

---

## 🔎 Exploratory Data Analysis

### 🔥 Correlation Matrix of Features

Understanding relationships between numeric features.

![Correlation Matrix](correlationmatrix.png)

---

### 📊 Distribution of CO₂ Emissions

The target variable shows a right-skewed distribution.

![CO2 Histogram](eda_histplot.png)

---

### 🧯 Fuel Consumption vs CO₂ Emissions (By Fuel Type)

Using dark mode plot to show relationships across fuel types.

![Fuel Comb vs CO2 Altair](model_altair.png)

---

### ⛽ Fuel Consumption vs CO₂ Emissions

- Fuel City vs CO₂  
  ![City vs CO2](scatplot1.png)

- Fuel Comb (L/100 km) vs CO₂  
  ![Comb vs CO2](scatplot2.png)

- Fuel Comb (mpg) vs CO₂  
  ![MPG vs CO2](scatplot3.png)

---

## 🧪 Model Training

Two models were tested and evaluated:

- **Polynomial Regression (degree=4)**
- **Spline Regression**

We used RMSE, MAE, and R² score for performance comparison.

---

### 📈 Model Comparison

Polynomial model outperformed Spline in all metrics.

![Model Comparison Chart](model_comparison_chart.png)

---

### 🔍 Polynomial Regression Fit (Zoomed View)

Fit comparison for degree 2, 3, and 4 polynomials.

![Poly Fit Zoom](polyinwork.png)

---

## 🛠️ Tech Stack

- **Language:** Python
- **Framework:** Streamlit
- **ML Libraries:** scikit-learn, pandas, numpy
- **Visualization:** seaborn, matplotlib, altair
- **Experiment Tracking:** Weights & Biases (wandb)
- **Deployment (Planned):** Streamlit Community Cloud / Docker

---

## 💻 Project Structure

```

CO2Emissions/
├── data/
├── models/
│   └── best\_model.pkl
├── notebooks/
├── results/
│   └── \*.png
├── source/
│   └── main.py
├── streamlitapp/
│   └── app.py
├── requirements.txt
├── README.md
└── ...

````

---

## 🚀 Running the App Locally

Install dependencies:

```bash
pip install -r requirements.txt
````

Launch the app:

```bash
streamlit run streamlitapp/app.py
```

---

## 🌐 Deployment Status

* ✅ GitHub repo: [CO2Emissions](https://github.com/Tamaghnatech/CO2Emissions)
* ⚠️ Deployment via Streamlit Cloud is under testing due to `requirements.txt` issues
* 🛠️ Docker support and Hugging Face Spaces deployment being considered

---

## 🧾 License

MIT License. Open for educational and research use. Credit the author if reused.

---

## 🤝 Contribute

PRs are welcome. If you'd like to contribute new model variants or improve UI, drop an issue or contact me.

```

---

✅ Let me know if you want this auto-committed to GitHub or rendered as a markdown preview.  
We can also generate a `Dockerfile`, fix the requirements for Streamlit Cloud, or even push this to Hugging Face Spaces if you want a zero-cost deployment option.
```
