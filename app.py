from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import datetime
import random
import requests
import hashlib
from user_agents import parse
import numpy as np
import pandas as pd  # Add pandas import
from dataclasses import dataclass
from typing import Optional, Dict, Any
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
import pickle
from preprocessing import preprocess_transaction_data, format_transaction_for_display
# from dotenv import load_dotenv

# Load environment variables
# load_dotenv()

# Configure Gemini

model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)

# In-memory storage for transactions (replace with database in production)
transactions = []

# Load trained models
MODELS_DIR = 'models'
ieee_model = joblib.load(os.path.join(MODELS_DIR, 'ieee_model.joblib')) if os.path.exists(os.path.join(MODELS_DIR, 'ieee_model.joblib')) else None
paysim_model = joblib.load(os.path.join(MODELS_DIR, 'paysim_model.joblib')) if os.path.exists(os.path.join(MODELS_DIR, 'paysim_model.joblib')) else None

# Load the label encoders
with open('C:/Users/asus/OneDrive/Desktop/Nirma/datasets/SAMD (Money_Laundering)/le_sender.pkl', 'rb') as f:
    le_sender = pickle.load(f)

with open('C:/Users/asus/OneDrive/Desktop/Nirma/datasets/SAMD (Money_Laundering)/le_receiver.pkl', 'rb') as f:
    le_receiver = pickle.load(f)

@dataclass
class CommonTransaction:
    """Common transaction class that standardizes fields from different datasets"""
    transaction_id: str
    timestamp: int
    amount: float
    transaction_type: str
    
    # Common fields from both datasets
    sender_id: Optional[str] = None
    receiver_id: Optional[str] = None
    sender_balance: Optional[float] = None
    receiver_balance: Optional[float] = None
    sender_previous_transactions: Optional[int] = None
    receiver_previous_transactions: Optional[int] = None
    
    # IEEE CIS specific fields
    card_number: Optional[str] = None
    card_type: Optional[str] = None
    card_bin: Optional[int] = None
    card_issuer: Optional[str] = None
    card_network: Optional[str] = None
    device_type: Optional[str] = None
    device_info: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    
    # PaySim specific fields
    payment_type: Optional[str] = None
    old_balance: Optional[float] = None
    new_balance: Optional[float] = None
    error_originating: Optional[str] = None
    error_destination: Optional[str] = None
    
    def to_ieee_features(self) -> np.ndarray:
        """Convert transaction to IEEE CIS feature vector"""
        features = [
            self.amount,
            self.sender_previous_transactions or 0,
            self.receiver_previous_transactions or 0,
            self.sender_balance or 0,
            self.receiver_balance or 0,
            # Add more IEEE specific features
        ]
        return np.array(features).reshape(1, -1)
    
    def to_paysim_features(self) -> np.ndarray:
        """Convert transaction to PaySim feature vector"""
        features = [
            self.amount,
            self.old_balance or 0,
            self.new_balance or 0,
            self.sender_previous_transactions or 0,
            self.receiver_previous_transactions or 0,
            # Add more PaySim specific features
        ]
        return np.array(features).reshape(1, -1)

def generate_transaction(transaction_amt, card_number, user_agent_string, transaction_type):
    """Generate realistic synthetic transaction data from minimal inputs."""

    # 🗓 Generate Transaction Timestamp
    transaction_dt = int(datetime.datetime.now().timestamp())

    # 🌍 Get Location from IP
    try:
        ip_info = requests.get("https://ipinfo.io/json").json()
        addr1 = ip_info.get("city", "Unknown")
        addr2 = ip_info.get("region", "Unknown")
        dist1 = random.randint(1, 50)  # Approximate distance
    except:
        addr1, addr2, dist1 = "Unknown", "Unknown", 0

    # 💳 Card Details
    card1 = int(card_number[:4])  # First 4 digits
    card2 = int(card_number[4:8])  # Middle 4 digits
    card3 = random.randint(100, 999)  # Synthetic CVV
    card4 = "Visa" if card1 in [4111, 4556, 4000] else "Mastercard"  # Card issuer
    card5 = random.randint(1, 999)  # Synthetic card bin
    card6 = random.choice(["debit", "credit"])

    # 📩 Email Domains
    email_domains = ["gmail.com", "yahoo.com", "outlook.com", "protonmail.com"]
    P_emaildomain = random.choice(email_domains)
    R_emaildomain = random.choice(email_domains)

    # 🖥 Device Info
    user_agent = parse(user_agent_string)
    DeviceType = user_agent.device.family
    DeviceInfo = f"{user_agent.browser.family} {user_agent.os.family}"

    # 🛒 Product Type
    ProductCD = random.choice(["W", "C", "R", "H", "S"])  # Based on dataset

    # 🔢 Synthetic Identity Features
    id_fields = {f"id_{i:02d}": random.randint(0, 10) for i in range(1, 39)}

    # 🔄 Synthetic Transaction Features
    V_fields = {f"V{i}": random.uniform(0, 1) for i in range(1, 340)}

    # 🏦 Generate Hash for Transaction ID
    TransactionID = hashlib.sha1(f"{transaction_amt}{card_number}{transaction_dt}".encode()).hexdigest()[:10]

    # 📦 Final Transaction Data
    transaction_data = {
        "TransactionID": TransactionID,
        "TransactionDT": transaction_dt,
        "TransactionAmt": transaction_amt,
        "ProductCD": ProductCD,
        "card1": card1, "card2": card2, "card3": card3, "card4": card4, "card5": card5, "card6": card6,
        "addr1": addr1, "addr2": addr2, "dist1": dist1,
        "P_emaildomain": P_emaildomain, "R_emaildomain": R_emaildomain,
        "DeviceType": DeviceType, "DeviceInfo": DeviceInfo,
        **id_fields, **V_fields  # Merge identity and V features
    }

    return transaction_data

def predict_fraud(transaction: CommonTransaction) -> Dict[str, Any]:
    """Predict fraud using appropriate model based on transaction type"""
    try:
        if transaction.transaction_type in ['online_purchase', 'card_present']:
            # Use IEEE CIS model for card-based transactions
            if ieee_model is not None:
                features = transaction.to_ieee_features()
                fraud_probability = ieee_model.predict_proba(features)[0][1]
                model_name = "IEEE CIS"
            else:
                fraud_probability = random.uniform(0, 1)
                model_name = "IEEE CIS (Fallback)"
        else:
            # Use PaySim model for other transaction types
            if paysim_model is not None:
                features = transaction.to_paysim_features()
                fraud_probability = paysim_model.predict_proba(features)[0][1]
                model_name = "PaySim"
            else:
                fraud_probability = random.uniform(0, 1)
                model_name = "PaySim (Fallback)"
        
        risk_level = "High" if fraud_probability > 0.7 else "Medium" if fraud_probability > 0.3 else "Low"
        
        return {
            "probability": float(fraud_probability),
            "level": risk_level,
            "model_used": model_name,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "probability": random.uniform(0, 1),
            "level": "Unknown",
            "model_used": "Fallback",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }

@app.route('/')
def onboarding():
    return render_template('onboarding.html')

@app.route('/select_role', methods=['POST'])
def select_role():
    role = request.form.get('role')
    if role == 'admin':
        # Set admin session data
        session['user'] = 'admin'  # Changed from 'admin_id' to 'user'
        session['role'] = 'admin'
        return redirect(url_for('admin_dashboard'))
    elif role == 'user':
        # Set user session data
        session['user'] = 'user'  # Changed from 'user_id' to 'user'
        session['role'] = 'user'
        return redirect(url_for('user_dashboard'))
    return redirect(url_for('onboarding'))

@app.route('/user_dashboard')
def user_dashboard():
    if 'user' not in session:
        return redirect(url_for('onboarding'))
    
    # Get current date and time
    current_datetime = datetime.datetime.now()
    
    # Convert time to seconds (seconds since midnight)
    time_in_seconds = current_datetime.hour * 3600 + current_datetime.minute * 60 + current_datetime.second
    
    # Format date as year, month, day
    formatted_date = {
        'year': current_datetime.year,
        'month': current_datetime.month,
        'day': current_datetime.day
    }
    
    # Format the date for display
    display_date = current_datetime.strftime("%Y-%m-%d")
    display_time = current_datetime.strftime("%H:%M:%S")
    
    return render_template('user_dashboard.html', 
                         current_date=display_date,
                         current_time=display_time,
                         time_in_seconds=time_in_seconds,
                         formatted_date=formatted_date)

@app.route('/admin_dashboard')
def admin_dashboard():
    # Check if user is logged in and is an admin
    if 'user' not in session:
        flash('Please login to access the dashboard', 'error')
        return redirect(url_for('login'))
    
    if session.get('role') != 'admin':
        flash('Admin access required', 'error')
        return redirect(url_for('login'))
    
    # Get filter parameters
    selected_dataset = request.args.get('dataset', 'paysim')
    search_term = request.args.get('search', '').lower()
    status_filter = request.args.get('status', 'all')
    date_sort = request.args.get('sort', 'newest')
    
    print(f"Selected dataset: {selected_dataset}")  # Debug print
    print(f"Search term: {search_term}")  # Debug print
    print(f"Status filter: {status_filter}")  # Debug print
    print(f"Date sort: {date_sort}")  # Debug print
    
    # Initialize variables
    paysim_data = []
    banking_data = []
    total_transactions = 0
    total_fraud_cases = 0
    success_rate = 0
    
    try:
        # Load both datasets first to calculate totals
        paysim_df = pd.read_csv('datasets/PaySim/final_paysim_dataset_with_explanations.csv')
        banking_df = pd.read_csv(r"C:\Users\asus\OneDrive\Desktop\Nirma\datasets\SAMD (Money_Laundering)\final samd dataset.csv")
        
        # Calculate total statistics
        total_transactions = len(paysim_df) + len(banking_df)
        total_fraud_cases = len(paysim_df[paysim_df['isFraud'] == 1]) + len(banking_df[banking_df['Is_laundering'] == 1])
        success_rate = ((total_transactions - total_fraud_cases) / total_transactions) * 100 if total_transactions > 0 else 0
        
        print(f"Total transactions across both datasets: {total_transactions}")
        print(f"Total fraud cases across both datasets: {total_fraud_cases}")
        print(f"Overall success rate: {success_rate:.2f}%")
        
        if selected_dataset == 'paysim':
            # Process PaySim data for display
            for _, row in paysim_df.iterrows():
                transaction = {
                    'transaction_id': row['step'],
                    'user': row['nameOrig'],
                    'amount': f"${row['amount']:,.2f}",
                    'payment_type': row['type'],
                    'recipient': row['nameDest'],
                    'old_balance': f"${row['oldbalanceOrg']:,.2f}",
                    'new_balance': f"${row['newbalanceOrig']:,.2f}",
                    'status': 'Completed' if row['isFraud'] == 0 else 'Failed',
                    'explanation': row['explanations']
                }
                
                # Apply filters
                if search_term and search_term not in str(transaction['user']).lower():
                    continue
                if status_filter == 'fraud' and transaction['status'] != 'Failed':
                    continue
                if status_filter == 'non-fraud' and transaction['status'] != 'Completed':
                    continue
                
                paysim_data.append(transaction)
            
            # Limit to first 100 rows after filtering
            paysim_data = paysim_data[:100]
            
            # Store in session for transaction details
            session['paysim_data'] = paysim_data
            
        elif selected_dataset == 'banking':
            # Process SAMD data for display
            for index, row in banking_df.iterrows():
                # Determine payment type from boolean columns
                payment_type = "Unknown"
                if row['Payment_type_Cash Deposit'] == 1:
                    payment_type = "Cash Deposit"
                elif row['Payment_type_Cash Withdrawal'] == 1:
                    payment_type = "Cash Withdrawal"
                elif row['Payment_type_Cheque'] == 1:
                    payment_type = "Cheque"
                elif row['Payment_type_Credit card'] == 1:
                    payment_type = "Credit Card"
                elif row['Payment_type_Cross-border'] == 1:
                    payment_type = "Cross-border"
                elif row['Payment_type_Debit card'] == 1:
                    payment_type = "Debit Card"
                
                # Format date from year, month, day columns
                date = f"{row['Year']}-{row['Month']:02d}-{row['Day']:02d}"
                
                transaction = {
                    'transaction_id': str(index + 1),  # Use 1-based index as transaction ID
                    'user': row['Sender_account'],
                    'amount': f"${row['Amount']:,.2f}",
                    'payment_type': payment_type,
                    'date': date,
                    'status': 'Completed' if row['Is_laundering'] == 0 else 'Failed',
                    'recipient': row['Receiver_account'],
                    'laundering_type': row['Laundering_type'],
                    'sender_info': row['Sender_Info'],
                    'receiver_info': row['Receiver_Info'],
                    'explanation': row['Model_Explanation']
                }
                
                # Apply filters
                if search_term and search_term not in str(transaction['user']).lower() and search_term not in str(transaction['recipient']).lower():
                    continue
                if status_filter == 'fraud' and transaction['status'] != 'Failed':
                    continue
                if status_filter == 'non-fraud' and transaction['status'] != 'Completed':
                    continue
                
                banking_data.append(transaction)
            
            # Sort by date
            banking_data.sort(key=lambda x: x['date'], reverse=(date_sort == 'newest'))
            
            # Limit to first 100 rows after filtering
            banking_data = banking_data[:100]
            
            # Store in session for transaction details
            session['banking_data'] = banking_data
            
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")  # Debug print
        flash(f'Error loading dataset: {str(e)}', 'error')
    
    return render_template('admin_dashboard.html',
                         selected_dataset=selected_dataset,
                         paysim_data=paysim_data,
                         banking_data=banking_data,
                         total_transactions=total_transactions,
                         fraud_cases=total_fraud_cases,
                         success_rate=success_rate)

@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    if session.get('role') != 'user':
        return redirect(url_for('onboarding'))
    
    try:
        # Get current date and time
        current_datetime = datetime.datetime.now()
        
        # Convert time to seconds (seconds since midnight)
        time_in_seconds = current_datetime.hour * 3600 + current_datetime.minute * 60 + current_datetime.second
        
        # Format date as year, month, day
        formatted_date = {
            'year': current_datetime.year,
            'month': current_datetime.month,
            'day': current_datetime.day
        }
        
        transaction_data = {
            'type': request.form.get('transaction_type'),
            'amount': float(request.form.get('amount')),
            'description': request.form.get('description'),
            'date': formatted_date,
            'time_in_seconds': time_in_seconds,
            'status': 'completed',
            'user_id': session.get('user'),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Add transaction to in-memory list
        transactions.append(transaction_data)
        flash('Transaction added successfully!', 'success')
    except Exception as e:
        flash('Error adding transaction. Please try again.', 'error')
        print(f"Error: {str(e)}")
    
    return redirect(url_for('user_dashboard'))

@app.route('/ieee_detect', methods=['POST'])
def ieee_detect():
    try:
        # Get data from request
        data = request.get_json()
        transaction_amt = float(data.get('amount', 0))
        card_number = data.get('card_number', '')
        user_agent_string = request.headers.get('User-Agent', '')
        transaction_type = data.get('type', 'online_purchase')

        # Generate transaction data
        transaction_data = generate_transaction(
            transaction_amt=transaction_amt,
            card_number=card_number,
            user_agent_string=user_agent_string,
            transaction_type=transaction_type
        )

        # Create CommonTransaction object
        common_transaction = CommonTransaction(
            transaction_id=transaction_data['TransactionID'],
            timestamp=transaction_data['TransactionDT'],
            amount=transaction_data['TransactionAmt'],
            transaction_type=transaction_type,
            card_number=card_number,
            card_type=transaction_data['card6'],
            card_issuer=transaction_data['card4'],
            device_type=transaction_data['DeviceType'],
            device_info=transaction_data['DeviceInfo'],
            location={
                'city': transaction_data['addr1'],
                'region': transaction_data['addr2'],
                'distance': transaction_data['dist1']
            }
        )

        # Get fraud prediction
        fraud_assessment = predict_fraud(common_transaction)

        # Prepare response
        response = {
            "status": "success",
            "transaction": transaction_data,
            "fraud_assessment": fraud_assessment
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        try:
            # Store message in in-memory list
            contact_message = {
                'name': name,
                'email': email,
                'subject': subject,
                'message': message,
                'timestamp': datetime.datetime.now().isoformat()
            }
            transactions.append(contact_message)
            
            flash('Message sent successfully!', 'success')
        except Exception as e:
            flash('Error sending message. Please try again.', 'error')
            print(f"Error: {str(e)}")
        
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

@app.route('/set_session', methods=['POST'])
def set_session():
    data = request.get_json()
    session['user_id'] = data.get('user_id')
    session['role'] = data.get('role')
    session['name'] = data.get('name')
    return jsonify({'status': 'success'})

def enhance_explanation(original_explanation):
    """Enhance the explanation using Gemini model"""
    try:
        # Create a prompt for Gemini
        prompt = f"""Given this transaction explanation: "{original_explanation}"

Please provide a detailed, believable explanation (4-5 sentences) that:
1. Describes the specific transaction pattern that triggered the alert
2. Includes exact amounts and currency (e.g., "unusual $45,000 transfer")
3. Mentions specific countries or regions involved
4. Explains why this pattern is suspicious (e.g., "unusual for this account's normal behavior")
5. Provides historical context (e.g., "first transaction of this type in 6 months")
6. Suggests the risk level (High/Medium/Low) with specific reasons
7. Includes any relevant regulatory concerns or compliance issues

Make the explanation sound like it's coming from a financial fraud analyst. Use specific details and professional terminology while keeping it clear and actionable."""

        # Get enhanced explanation from Gemini
        response = model.generate_content(prompt)
        enhanced_explanation = response.text.strip()
        
        # Add a separator line and format the original explanation
        # enhanced_explanation = f"{enhanced_explanation}\n\n---\n\nTechnical Details:\n{original_explanation}"
        
        return enhanced_explanation
    except Exception as e:
        print(f"Error enhancing explanation: {str(e)}")
        return original_explanation

@app.route('/transaction_details/<transaction_id>')
def transaction_details(transaction_id):
    if 'user' not in session or session['role'] != 'admin':
        flash('Please login as admin to view transaction details.', 'error')
        return redirect(url_for('login'))
    
    selected_dataset = request.args.get('dataset', 'paysim')
    transaction = None
    
    try:
        if selected_dataset == 'paysim':
            # Load PaySim dataset directly
            df = pd.read_csv('datasets/PaySim/final_paysim_dataset_with_explanations.csv')
            row = df[df['step'] == int(transaction_id)]
            if not row.empty:
                transaction = {
                    'transaction_id': row['step'].iloc[0],
                    'user': row['nameOrig'].iloc[0],
                    'amount': f"${row['amount'].iloc[0]:,.2f}",
                    'payment_type': row['type'].iloc[0],
                    'recipient': row['nameDest'].iloc[0],
                    'old_balance': f"${row['oldbalanceOrg'].iloc[0]:,.2f}",
                    'new_balance': f"${row['newbalanceOrig'].iloc[0]:,.2f}",
                    'status': 'Completed' if row['isFraud'].iloc[0] == 0 else 'Failed',
                    'explanation': row['explanations'].iloc[0]
                }
                # Enhance the explanation using Gemini
                transaction['explanation'] = enhance_explanation(transaction['explanation'])
                return render_template('paysim_details.html', transaction=transaction)
        else:
            # Load SAMD dataset directly
            df = pd.read_csv(r"C:\Users\asus\OneDrive\Desktop\Nirma\datasets\SAMD (Money_Laundering)\final samd dataset.csv")
            # Find the transaction by index since we're using 1-based index IDs
            try:
                row = df.iloc[int(transaction_id) - 1]  # Convert to 0-based index
            except (IndexError, ValueError) as e:
                print(f"Error finding transaction: {str(e)}")
                flash('Transaction not found.', 'error')
                return redirect(url_for('admin_dashboard'))
            
            # Determine payment type from boolean columns
            payment_type = "Unknown"
            if row['Payment_type_Cash Deposit'] == 1:
                payment_type = "Cash Deposit"
            elif row['Payment_type_Cash Withdrawal'] == 1:
                payment_type = "Cash Withdrawal"
            elif row['Payment_type_Cheque'] == 1:
                payment_type = "Cheque"
            elif row['Payment_type_Credit card'] == 1:
                payment_type = "Credit Card"
            elif row['Payment_type_Cross-border'] == 1:
                payment_type = "Cross-border"
            elif row['Payment_type_Debit card'] == 1:
                payment_type = "Debit Card"
            
            # Format date from year, month, day columns
            date = f"{row['Year']}-{row['Month']:02d}-{row['Day']:02d}"
            
            transaction = {
                'transaction_id': transaction_id,
                'user': row['Sender_account'],
                'amount': f"${row['Amount']:,.2f}",
                'payment_type': payment_type,
                'date': date,
                'recipient': row['Receiver_account'],
                'status': 'Completed' if row['Is_laundering'] == 0 else 'Failed',
                'laundering_type': row['Laundering_type'],
                'sender_info': row['Sender_Info'],
                'receiver_info': row['Receiver_Info'],
                'explanation': row['Model_Explanation'] if 'Model_Explanation' in row else "No explanation available"
            }
            # Enhance the explanation using Gemini
            transaction['explanation'] = enhance_explanation(transaction['explanation'])
            return render_template('money_laundering_details.html', transaction=transaction)
    
    except Exception as e:
        print(f"Error loading transaction details: {str(e)}")
        flash('Error loading transaction details.', 'error')
        return redirect(url_for('admin_dashboard'))
    
    if not transaction:
        flash('Transaction not found.', 'error')
        return redirect(url_for('admin_dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # For demo purposes, accept any username with password 'admin123'
        if password == 'admin123':
            session['user'] = username
            session['role'] = 'admin'
            print(f"Login successful for user: {username}")
            print(f"Session data: {session}")
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid credentials', 'error')
            print("Login failed - invalid credentials")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    # Clear all session data
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/encode_info', methods=['POST'])
def encode_info():
    try:
        data = request.get_json()
        sender_info = data.get('sender_info')
        receiver_info = data.get('receiver_info')
        
        if not sender_info or not receiver_info:
            return jsonify({'error': 'Sender and receiver information are required'}), 400
            
        # Validate that the info strings contain the expected format (location - currency)
        if '-' not in sender_info or '-' not in receiver_info:
            return jsonify({'error': 'Invalid format. Expected format: "location - currency"'}), 400
            
        try:
            # Attempt to encode the sender and receiver info
            sender_encoded = le_sender.transform([sender_info])[0]
            receiver_encoded = le_receiver.transform([receiver_info])[0]
            
            return jsonify({
                'sender_encoded': int(sender_encoded),
                'receiver_encoded': int(receiver_encoded)
            })
        except ValueError as e:
            # Handle unknown labels
            return jsonify({
                'error': f'Unknown location-currency combination. Please check the values and try again.',
                'details': str(e)
            }), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Format the prompt with the transaction data
        prompt = f"""
        Analyze this bank transaction for potential fraud:
        
        Transaction Details:
        - Time: {data['Time']} seconds since midnight
        - Amount: ${data['Amount']}
        - Sender Account: {data['Sender_account']}
        - Receiver Account: {data['Receiver_account']}
        - Sender Info (Location - Currency): {data['Sender_Info']}
        - Receiver Info (Location - Currency): {data['Receiver_Info']}
        - Payment Type Flags: {', '.join(k for k, v in data.items() if k.startswith('Payment_type_') and v)}
        
        Additional Context:
        - Days Since Start: {data['Days_Since_Start']}
        - Date: {data['Year']}-{data['Month']}-{data['Day']}
        
        Please analyze this transaction and:
        1. Assess the fraud risk (High, Medium, Low)
        2. Explain the key risk factors or suspicious patterns if any
        3. Provide recommendations for additional verification if needed
        """
        
        # Get prediction from Gemini
        response = model.generate_content(prompt)
        
        return jsonify({
            'prediction': response.text
        })
        
    except Exception as e:
        print(f"Error in predict route: {str(e)}")  # Add logging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 