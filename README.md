AI Clinical Triage & Disease Diagnostic Engine

Overview

This project is a fully distributed Web Application that utilizes a Type-1 Mamdani Fuzzy Inference System (FIS) to diagnose clinical respiratory and viral diseases based on patient vitals and symptoms.

Unlike traditional binary logic models, this engine processes medical ambiguity using continuous fuzzy sets, incorporating a custom 100% Coverage Core Vitals Matrix to ensure flawless clinical triaging even with incomplete data.

Engineering Architecture

Frontend: HTML5, Tailwind CSS, Client-Side PDF Generation (html2pdf.js).

Backend: Python Flask API.

Math Engine: pyit2fls, numpy (Trapezoidal Membership Functions, Centroid Defuzzification).

Core Medical Parameters

The engine evaluates 9 clinical inputs:

Body Temperature (°F)

Blood Oxygen (SpO2 %) (Custom injected parameter for respiratory scaling)

Respiratory Rate (breaths/min)

Headache Severity (0-10)

Cough Severity (0-10)

Sore Throat (Boolean)

Flu Symptoms (Boolean)

Vomiting (Boolean)

Diarrhea (Boolean)

Target Outputs

Normal / Healthy

General Viral / Flu (Broad Triage)

Coronavirus

Pneumonia

Typhoid

Malaria

Installation & Execution

Clone the repository.

Install the mathematical dependencies:

pip install -r requirements.txt


Run the Flask server:

python app.py


Access the web interface at http://127.0.0.1:5000.