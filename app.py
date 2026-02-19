from flask import Flask, request, jsonify, render_template_string
from pyit2fls import trapezoid_mf, T1FS, T1Mamdani
from numpy import linspace
import numpy as np
import traceback

app = Flask(__name__)

# ==========================================
# PHASE 1: THE MATHEMATICAL ENGINE (BACKEND)
# ==========================================
def evaluate_disease_fuzzy(fever, headache, rrate, cough, sthroat, flu, vomit, diarr, oxygen):
    # 1. Universes of Discourse (Fixed Medical Data)
    d_fever = linspace(98, 104, 500)
    d_headache = linspace(0, 10, 500)
    d_rrate = linspace(10, 40, 500) 
    d_cough = linspace(0, 10, 500)
    d_sthroat = linspace(0, 1, 500)
    d_flu = linspace(0, 1, 500)
    d_vomit = linspace(0, 1, 500)
    d_diarr = linspace(0, 1, 500)
    d_oxygen = linspace(70, 100, 500) 
    d_disease = linspace(0, 100, 500)

    # 2. Fuzzy Sets
    Fever_low = T1FS(d_fever, trapezoid_mf, [97.9, 98, 98.5, 99, 1])
    Fever_med = T1FS(d_fever, trapezoid_mf, [98.5, 99, 101.5, 102, 1])
    Fever_high = T1FS(d_fever, trapezoid_mf, [101.5, 102, 104, 104.1, 1])

    Headache_mod = T1FS(d_headache, trapezoid_mf, [-0.1, 0, 4, 7, 1])
    Headache_sev = T1FS(d_headache, trapezoid_mf, [4, 7, 10, 10.1, 1])

    Rr_low = T1FS(d_rrate, trapezoid_mf, [9.9, 10, 16, 20, 1]) 
    Rr_med = T1FS(d_rrate, trapezoid_mf, [16, 20, 25, 30, 1])  
    Rr_high = T1FS(d_rrate, trapezoid_mf, [25, 30, 40, 40.1, 1]) 

    Cough_low = T1FS(d_cough, trapezoid_mf, [-0.1, 0, 3.5, 5, 1])
    Cough_high = T1FS(d_cough, trapezoid_mf, [4.5, 7, 10, 10.1, 1])

    Sthroat_low = T1FS(d_sthroat, trapezoid_mf, [-0.1, 0, 0.2, 0.4, 1])
    Sthroat_high = T1FS(d_sthroat, trapezoid_mf, [0.4, 0.6, 1, 1.1, 1])

    Flu_no = T1FS(d_flu, trapezoid_mf, [-0.1, 0, 0.4, 0.5, 1])
    Flu_yes = T1FS(d_flu, trapezoid_mf, [0.4, 0.5, 1, 1.1, 1])

    Vomit_no = T1FS(d_vomit, trapezoid_mf, [-0.1, 0, 0.4, 0.5, 1])
    Vomit_yes = T1FS(d_vomit, trapezoid_mf, [0.4, 0.5, 1, 1.1, 1])

    Diarr_no = T1FS(d_diarr, trapezoid_mf, [-0.1, 0, 0.4, 0.5, 1])
    Diarr_yes = T1FS(d_diarr, trapezoid_mf, [0.4, 0.5, 1, 1.1, 1])

    Ox_low = T1FS(d_oxygen, trapezoid_mf, [69.9, 70, 85, 92, 1]) 
    Ox_normal = T1FS(d_oxygen, trapezoid_mf, [90, 95, 100, 100.1, 1]) 

    # RESTRUCTURED OUTPUT UNIVERSE
    Di_normal = T1FS(d_disease, trapezoid_mf, [-0.1, 0, 10, 15, 1])
    Di_viral = T1FS(d_disease, trapezoid_mf, [15, 20, 25, 30, 1]) 
    Di_corona = T1FS(d_disease, trapezoid_mf, [30, 35, 45, 50, 1])
    Di_pneumo = T1FS(d_disease, trapezoid_mf, [50, 55, 65, 70, 1])
    Di_typhoid = T1FS(d_disease, trapezoid_mf, [70, 75, 80, 85, 1])
    Di_malaria = T1FS(d_disease, trapezoid_mf, [85, 90, 100, 100.1, 1])

    SYS = T1Mamdani()
    SYS.add_input_variable("Fever")
    SYS.add_input_variable("Headache")
    SYS.add_input_variable("RespRate")
    SYS.add_input_variable("Cough")
    SYS.add_input_variable("SoreThroat")
    SYS.add_input_variable("Flu")
    SYS.add_input_variable("Vomit")
    SYS.add_input_variable("Diarrhea")
    SYS.add_input_variable("Oxygen") 
    SYS.add_output_variable("Disease")

    # ==========================================================
    # 4. RULE BASE: 100% COVERAGE MATRIX & EXPERT SYSTEM
    # ==========================================================
    
    # 1. Normal Oxygen Scenarios
    SYS.add_rule([("Fever", Fever_low), ("Oxygen", Ox_normal)], [("Disease", Di_normal)])
    SYS.add_rule([("Fever", Fever_med), ("Oxygen", Ox_normal), ("Diarrhea", Diarr_no), ("Vomit", Vomit_no)], [("Disease", Di_viral)])
    SYS.add_rule([("Fever", Fever_high), ("Oxygen", Ox_normal), ("Diarrhea", Diarr_no), ("Vomit", Vomit_no)], [("Disease", Di_viral)])

    # 2. Low Oxygen Scenarios (Critical Respiratory)
    SYS.add_rule([("Fever", Fever_low), ("Oxygen", Ox_low)], [("Disease", Di_pneumo)])
    SYS.add_rule([("Fever", Fever_med), ("Oxygen", Ox_low)], [("Disease", Di_pneumo)])
    SYS.add_rule([("Fever", Fever_high), ("Oxygen", Ox_low)], [("Disease", Di_corona)])

    # 3. SPECIFIC DISEASE TARGETING
    SYS.add_rule([("RespRate", Rr_high), ("Cough", Cough_high), ("Oxygen", Ox_low), ("Flu", Flu_no)], [("Disease", Di_pneumo)])
    SYS.add_rule([("Fever", Fever_high), ("Diarrhea", Diarr_yes), ("Headache", Headache_sev)], [("Disease", Di_typhoid)])
    SYS.add_rule([("Fever", Fever_med), ("Vomit", Vomit_yes), ("Headache", Headache_sev), ("Oxygen", Ox_normal)], [("Disease", Di_malaria)])
    SYS.add_rule([("Fever", Fever_high), ("Vomit", Vomit_yes), ("Oxygen", Ox_normal)], [("Disease", Di_malaria)])
    SYS.add_rule([("Cough", Cough_high), ("Flu", Flu_yes), ("Oxygen", Ox_low)], [("Disease", Di_corona)])

    # 5. Execution & Safe Defuzzification
    _, tr = SYS.evaluate({
        "Fever": fever, "Headache": headache, "RespRate": rrate, "Cough": cough,
        "SoreThroat": sthroat, "Flu": flu, "Vomit": vomit, "Diarrhea": diarr, "Oxygen": oxygen
    })

    disease_out = tr.get("Disease", 0.0)
    
    if isinstance(disease_out, (int, float, np.float64, np.float32)):
        crisp_score = float(disease_out)
    elif hasattr(disease_out, 'mf'):
        mf_values = np.array(disease_out.mf)
        if np.sum(mf_values) == 0:
            crisp_score = 0.0
        else:
            crisp_score = float(np.sum(d_disease * mf_values) / np.sum(mf_values))
    else:
        crisp_score = 0.0

    if np.isnan(crisp_score):
        return 0.0, "Inconclusive (Symptoms don't match clinical rules)"

    if crisp_score < 0: crisp_score = 0
    if crisp_score > 100: crisp_score = 100

    if 0 <= crisp_score < 15: label = "Normal"
    elif 15 <= crisp_score < 30: label = "General Viral / Flu"
    elif 30 <= crisp_score < 50: label = "Coronavirus"
    elif 50 <= crisp_score < 70: label = "Pneumonia"
    elif 70 <= crisp_score < 85: label = "Typhoid"
    elif 85 <= crisp_score <= 100: label = "Malaria"
    else: label = "Unknown"

    return crisp_score, label

# ==========================================
# PHASE 2: THE API AND FRONTEND (WEB)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Diagnostic System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
    <style>
        /* Smooth scrolling for the UI */
        html { scroll-behavior: smooth; }
    </style>
</head>
<body class="bg-gray-900 text-white font-sans min-h-screen relative p-8">

    <!-- FULL SCREEN PDF GENERATION OVERLAY -->
    <div id="pdfLoadingOverlay" class="fixed inset-0 bg-gray-900 z-[10000] hidden flex-col items-center justify-center">
        <svg class="animate-spin h-16 w-16 text-blue-500 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
        </svg>
        <h2 class="text-3xl font-bold text-white">Generating Clinical Report...</h2>
        <p class="text-gray-400 mt-2">Mathematically rendering high-resolution document.</p>
    </div>

    <!-- MAIN WEBSITE UI -->
    <div id="mainUI" class="max-w-4xl mx-auto bg-gray-800 rounded-xl shadow-2xl overflow-hidden border border-gray-700 relative z-10">
        <div class="bg-blue-600 p-6 flex justify-between items-center">
            <div>
                <h1 class="text-3xl font-bold">Fuzzy Logic Disease Diagnostic Engine</h1>
                <p class="text-blue-200 mt-2">Enter patient demographics and symptoms to evaluate clinical risk.</p>
            </div>
        </div>
        
        <div class="p-8 grid grid-cols-1 md:grid-cols-2 gap-8">
            <!-- Form -->
            <form id="diagForm" class="space-y-4">
                
                <!-- NEW: PATIENT DEMOGRAPHICS SECTION -->
                <div class="bg-gray-700 p-4 rounded-lg border border-gray-600 mb-6">
                    <h3 class="text-lg font-bold text-blue-400 border-b border-gray-600 pb-2 mb-4">1. Patient Demographics</h3>
                    <div class="space-y-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-300">Full Name</label>
                            <input type="text" id="patName" placeholder="e.g. Jane Doe" class="mt-1 block w-full bg-gray-800 border border-gray-600 rounded-md py-2 px-3 text-white focus:outline-none focus:border-blue-500">
                        </div>
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm font-medium text-gray-300">Age</label>
                                <input type="number" id="patAge" placeholder="e.g. 45" class="mt-1 block w-full bg-gray-800 border border-gray-600 rounded-md py-2 px-3 text-white focus:outline-none focus:border-blue-500">
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-300">Gender</label>
                                <select id="patGender" class="mt-1 block w-full bg-gray-800 border border-gray-600 rounded-md py-2 px-3 text-white focus:outline-none focus:border-blue-500">
                                    <option value="Not Specified">Select...</option>
                                    <option value="Male">Male</option>
                                    <option value="Female">Female</option>
                                    <option value="Other">Other</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>

                <h3 class="text-lg font-bold text-blue-400 border-b border-gray-600 pb-2 mb-4">2. Clinical Vitals & Symptoms</h3>
                
                <div>
                    <label class="block text-sm font-medium text-gray-400">Body Temperature (°F) <span id="fv_val" class="text-white float-right">98.6</span></label>
                    <input type="range" id="fever" min="98" max="104" step="0.1" value="98.6" class="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer" oninput="document.getElementById('fv_val').innerText = this.value">
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-400">Blood Oxygen SpO2 (%) <span id="ox_val" class="text-white float-right">98</span></label>
                    <input type="range" id="oxygen" min="70" max="100" step="1" value="98" class="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer" oninput="document.getElementById('ox_val').innerText = this.value">
                </div>

                <div>
                    <label class="block text-sm font-medium text-gray-400">Respiratory Rate (breaths/min) <span id="rr_val" class="text-white float-right">16</span></label>
                    <input type="range" id="rrate" min="10" max="40" step="1" value="16" class="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer" oninput="document.getElementById('rr_val').innerText = this.value">
                </div>

                <div>
                    <label class="block text-sm font-medium text-gray-400">Headache Severity (0-10) <span id="hd_val" class="text-white float-right">0</span></label>
                    <input type="range" id="headache" min="0" max="10" step="1" value="0" class="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer" oninput="document.getElementById('hd_val').innerText = this.value">
                </div>

                <div>
                    <label class="block text-sm font-medium text-gray-400">Cough Severity (0-10) <span id="cg_val" class="text-white float-right">0</span></label>
                    <input type="range" id="cough" min="0" max="10" step="1" value="0" class="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer" oninput="document.getElementById('cg_val').innerText = this.value">
                </div>

                <div class="grid grid-cols-2 gap-4 pt-4 border-t border-gray-700">
                    <div>
                        <label class="block text-sm font-medium text-gray-400">Sore Throat</label>
                        <select id="sthroat" class="mt-1 block w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                            <option value="0">No</option><option value="1">Yes</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-400">Flu Symptoms</label>
                        <select id="flu" class="mt-1 block w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 focus:outline-none">
                            <option value="0">No</option><option value="1">Yes</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-400">Vomiting</label>
                        <select id="vomit" class="mt-1 block w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 focus:outline-none">
                            <option value="0">No</option><option value="1">Yes</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-400">Diarrhea</label>
                        <select id="diarr" class="mt-1 block w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 focus:outline-none">
                            <option value="0">No</option><option value="1">Yes</option>
                        </select>
                    </div>
                </div>

                <button type="button" onclick="runDiagnosis()" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg mt-6 transition duration-300 shadow-lg">
                    Run AI Diagnosis
                </button>
            </form>

            <!-- Results Panel -->
            <div class="bg-gray-900 rounded-xl p-6 border border-gray-700 flex flex-col justify-center items-center text-center">
                <svg class="w-16 h-16 text-gray-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"></path></svg>
                <h2 class="text-xl font-semibold text-gray-400">Diagnostic Result</h2>
                
                <div id="loading" class="hidden mt-4 text-blue-400">Computing Fuzzy Matrices...</div>
                
                <div id="resultBox" class="hidden mt-6 w-full flex flex-col items-center">
                    <p class="text-sm text-gray-400 uppercase tracking-wide">Detected Condition</p>
                    <p id="diseaseLabel" class="text-4xl font-bold text-white mt-1 uppercase text-center"></p>
                    
                    <div class="mt-6 w-full bg-gray-700 rounded-full h-4 overflow-hidden">
                        <div id="riskBar" class="bg-red-500 h-4 rounded-full transition-all duration-1000" style="width: 0%"></div>
                    </div>
                    <p class="text-right text-xs text-gray-400 mt-2 w-full">Crisp Risk Score: <span id="scoreVal"></span> / 100</p>
                    
                    <!-- Download Report Button -->
                    <button id="downloadBtn" onclick="generatePDF()" class="hidden mt-8 bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-6 rounded-lg transition duration-300 shadow-lg flex items-center">
                        <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                        Download Medical Report
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- THE PERFECT RENDER TEMPLATE (Redesigned to mirror the requested image) -->
    <div id="pdfTemplate" class="hidden bg-white font-sans mx-auto" style="width: 800px; max-width: 800px; box-sizing: border-box; background-color: #ffffff;">
        
        <!-- Header Section (Dark Blue) -->
        <div style="background-color: #2c3e50; color: #ffffff; padding: 30px; text-align: center;">
            <h1 style="margin: 0; font-size: 28px; font-weight: bold; letter-spacing: 1px;">CLINICAL HANDOFF REPORT</h1>
            <p style="margin: 8px 0 0 0; font-size: 12px; color: #cbd5e1;">AI Diagnostic Engine | Mamdani Fuzzy Inference System</p>
        </div>

        <!-- Main Content Area -->
        <div style="padding: 40px;">
            <h2 style="text-align: center; font-size: 22px; color: #1e293b; margin-top: 0; margin-bottom: 30px; font-weight: bold;">Clinical Triage Assessment</h2>

            <!-- I. Patient Information -->
            <h3 style="font-size: 18px; color: #334155; margin-bottom: 10px; font-weight: bold;">I. Patient Information</h3>
            <div style="background-color: #f8fafc; border-radius: 12px; padding: 20px; margin-bottom: 30px; border: 1px solid #e2e8f0;">
                <h4 style="font-size: 15px; color: #475569; margin: 0 0 15px 0; font-weight: bold;">1. Demographics</h4>
                <div style="display: flex; justify-content: space-between; font-size: 14px; color: #1e293b;">
                    <div style="width: 48%;">
                        <p style="margin: 8px 0;"><strong>Patient Name:</strong> <span id="repName"></span></p>
                        <p style="margin: 8px 0;"><strong>Age:</strong> <span id="repAge"></span></p>
                        <p style="margin: 8px 0;"><strong>Gender:</strong> <span id="repGender"></span></p>
                    </div>
                    <div style="width: 48%;">
                        <p style="margin: 8px 0;"><strong>Assessment Date:</strong> <span id="repDate"></span></p>
                        <p style="margin: 8px 0;"><strong>Record ID:</strong> <span id="repID"></span></p>
                    </div>
                </div>
            </div>

            <!-- II. Current Status (Vitals & Symptoms) -->
            <h3 style="font-size: 18px; color: #334155; margin-bottom: 10px; font-weight: bold;">II. Current Status</h3>
            <div style="background-color: #f8fafc; border-radius: 12px; padding: 20px; margin-bottom: 30px; border: 1px solid #e2e8f0;">
                
                <h4 style="font-size: 15px; color: #475569; margin: 0 0 15px 0; font-weight: bold;">1. Vital Signs</h4>
                <div style="display: flex; justify-content: space-between; font-size: 14px; color: #1e293b; margin-bottom: 20px;">
                    <div style="width: 48%;">
                        <p style="margin: 8px 0;"><strong>Temperature:</strong> <span id="repFever"></span> °F</p>
                        <p style="margin: 8px 0;"><strong>Respiratory Rate:</strong> <span id="repRrate"></span> breaths/min</p>
                    </div>
                    <div style="width: 48%;">
                        <p style="margin: 8px 0;"><strong>Oxygen Saturation:</strong> <span id="repOxygen"></span> % SpO2</p>
                    </div>
                </div>

                <div style="border-top: 1px solid #cbd5e1; margin: 15px 0;"></div>

                <h4 style="font-size: 15px; color: #475569; margin: 15px 0 15px 0; font-weight: bold;">2. Clinical Symptoms Evaluated</h4>
                <div style="display: flex; justify-content: space-between; font-size: 14px; color: #1e293b;">
                     <div style="width: 48%;">
                        <p style="margin: 8px 0;"><strong>Headache Severity:</strong> <span id="repHeadache"></span> / 10</p>
                        <p style="margin: 8px 0;"><strong>Cough Severity:</strong> <span id="repCough"></span> / 10</p>
                        <p style="margin: 8px 0;"><strong>Sore Throat:</strong> <span id="repSthroat"></span></p>
                    </div>
                    <div style="width: 48%;">
                        <p style="margin: 8px 0;"><strong>Flu Symptoms:</strong> <span id="repFlu"></span></p>
                        <p style="margin: 8px 0;"><strong>Vomiting:</strong> <span id="repVomit"></span></p>
                        <p style="margin: 8px 0;"><strong>Diarrhea:</strong> <span id="repDiarr"></span></p>
                    </div>
                </div>
            </div>

            <!-- III. AI Diagnostic Assessment -->
            <h3 style="font-size: 18px; color: #334155; margin-bottom: 10px; font-weight: bold;">III. AI Diagnostic Engine Assessment</h3>
             <div style="background-color: #f8fafc; border: 2px solid #cbd5e1; border-radius: 12px; padding: 25px; margin-top: 10px;">
                 <p style="font-size: 20px; color: #1e293b; margin: 0 0 12px 0;"><strong>Detected Condition:</strong> <span id="repDisease" style="color: #dc2626; font-weight: bold; text-transform: uppercase;"></span></p>
                 <p style="font-size: 16px; color: #1e293b; margin: 0;"><strong>Crisp Risk Probability:</strong> <span id="repScore" style="color: #2563eb; font-weight: bold;"></span> / 100.00</p>
             </div>
             
             <!-- Footer Disclaimer -->
             <div style="text-align: center; color: #94a3b8; font-size: 11px; margin-top: 50px; border-top: 1px solid #e2e8f0; padding-top: 20px;">
                 <p style="margin: 0 0 4px 0;"><strong>* DISCLAIMER:</strong> This document is generated algorithmically via a Type-1 Mamdani Fuzzy Inference System.</p>
                 <p style="margin: 0;">It is intended for academic and triage demonstration purposes only. Always consult a certified physician for final diagnosis. *</p>
             </div>
        </div>
    </div>

    <script>
        async function runDiagnosis() {
            const loadEl = document.getElementById('loading');
            loadEl.classList.remove('hidden');
            loadEl.innerHTML = "Computing Fuzzy Matrices...";
            loadEl.className = "mt-4 text-blue-400";
            document.getElementById('resultBox').classList.add('hidden');
            
            const sthroatText = document.getElementById('sthroat').options[document.getElementById('sthroat').selectedIndex].text;
            const fluText = document.getElementById('flu').options[document.getElementById('flu').selectedIndex].text;
            const vomitText = document.getElementById('vomit').options[document.getElementById('vomit').selectedIndex].text;
            const diarrText = document.getElementById('diarr').options[document.getElementById('diarr').selectedIndex].text;

            const payload = {
                fever: parseFloat(document.getElementById('fever').value),
                oxygen: parseFloat(document.getElementById('oxygen').value),
                rrate: parseFloat(document.getElementById('rrate').value),
                headache: parseFloat(document.getElementById('headache').value),
                cough: parseFloat(document.getElementById('cough').value),
                sthroat: parseFloat(document.getElementById('sthroat').value),
                flu: parseFloat(document.getElementById('flu').value),
                vomit: parseFloat(document.getElementById('vomit').value),
                diarr: parseFloat(document.getElementById('diarr').value)
            };

            // Grab patient demographic data (default to N/A if left blank by user)
            const pName = document.getElementById('patName').value || 'Not Provided';
            const pAge = document.getElementById('patAge').value || 'Not Provided';
            const pGender = document.getElementById('patGender').value;

            try {
                const response = await fetch('/api/diagnose', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                
                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.error || "Server Error " + response.status);
                }
                
                const data = await response.json();
                
                loadEl.classList.add('hidden');
                document.getElementById('resultBox').classList.remove('hidden');
                
                document.getElementById('diseaseLabel').innerText = data.disease;
                document.getElementById('scoreVal').innerText = data.crisp_score.toFixed(2);
                
                const bar = document.getElementById('riskBar');
                bar.style.width = data.crisp_score + '%';
                
                const btn = document.getElementById('downloadBtn');
                
                if(data.disease.includes('Inconclusive')) {
                    bar.className = 'bg-yellow-500 h-4 rounded-full transition-all duration-1000';
                    btn.classList.add('hidden'); 
                } else {
                    btn.classList.remove('hidden'); 
                    
                    if(data.disease === 'Normal') {
                        bar.className = 'bg-green-500 h-4 rounded-full transition-all duration-1000';
                    } else if(data.disease === 'General Viral / Flu') {
                        bar.className = 'bg-yellow-500 h-4 rounded-full transition-all duration-1000';
                    } else {
                        bar.className = 'bg-red-500 h-4 rounded-full transition-all duration-1000';
                    }

                    // Populate the hidden strict-layout PDF Template
                    document.getElementById('repName').innerText = pName;
                    document.getElementById('repAge').innerText = pAge;
                    document.getElementById('repGender').innerText = pGender;
                    document.getElementById('repID').innerText = "AI-" + Math.floor(Math.random() * 1000000);
                    
                    document.getElementById('repDate').innerText = new Date().toLocaleString();
                    document.getElementById('repFever').innerText = payload.fever;
                    document.getElementById('repOxygen').innerText = payload.oxygen;
                    document.getElementById('repRrate').innerText = payload.rrate;
                    document.getElementById('repHeadache').innerText = payload.headache;
                    document.getElementById('repCough').innerText = payload.cough;
                    document.getElementById('repSthroat').innerText = sthroatText;
                    document.getElementById('repFlu').innerText = fluText;
                    document.getElementById('repVomit').innerText = vomitText;
                    document.getElementById('repDiarr').innerText = diarrText;
                    
                    document.getElementById('repDisease').innerText = data.disease;
                    document.getElementById('repScore').innerText = data.crisp_score.toFixed(2);
                }

            } catch (err) {
                loadEl.innerHTML = `<span class="text-red-500 font-bold">Engine Error:</span> ${err.message}`;
            }
        }

        // ==========================================
        // THE BULLETPROOF RENDER TAKEOVER METHOD
        // ==========================================
        function generatePDF() {
            const template = document.getElementById('pdfTemplate');
            const mainUI = document.getElementById('mainUI');
            const overlay = document.getElementById('pdfLoadingOverlay');

            // 1. Show the Loading Overlay to hide the "glitch" from the user
            overlay.classList.remove('hidden');
            overlay.style.display = 'flex';

            // 2. Hide the Main UI and Unhide the PDF Template into the normal flow
            mainUI.classList.add('hidden');
            template.classList.remove('hidden');
            
            // Scroll to top to ensure canvas starts at 0,0
            window.scrollTo(0, 0);

            var opt = {
              margin:       [10, 0, 10, 0], // Top, Right, Bottom, Left margins
              filename:     'Clinical_Handoff_Report.pdf',
              image:        { type: 'jpeg', quality: 1.0 },
              html2canvas:  { scale: 2 },
              jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
            };

            // 3. Give the browser engine 200ms to natively paint the template
            setTimeout(() => {
                html2pdf().set(opt).from(template).save().then(function() {
                    // 4. Reverse everything back to normal
                    template.classList.add('hidden');
                    mainUI.classList.remove('hidden');
                    overlay.classList.add('hidden');
                    overlay.style.display = '';
                });
            }, 200);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    try:
        data = request.json
        score, disease = evaluate_disease_fuzzy(
            data['fever'], data['headache'], data['rrate'], data['cough'],
            data['sthroat'], data['flu'], data['vomit'], data['diarr'], data['oxygen']
        )
        return jsonify({"crisp_score": score, "disease": disease})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Math Exception: No rules triggered for these exact inputs or invalid fuzzy mapping. Details: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)