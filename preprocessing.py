import datetime
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any

# Load the label encoders
with open('C:/Users/asus/OneDrive/Desktop/Nirma/datasets/SAMD (Money_Laundering)/le_sender.pkl', 'rb') as f:
    le_sender = pickle.load(f)

with open('C:/Users/asus/OneDrive/Desktop/Nirma/datasets/SAMD (Money_Laundering)/le_receiver.pkl', 'rb') as f:
    le_receiver = pickle.load(f)

def preprocess_transaction_data(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess the transaction data for bank transfers.
    
    Args:
        form_data (Dict[str, Any]): Dictionary containing form data
        
    Returns:
        Dict[str, Any]: Preprocessed transaction data
    """
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
    
    # Calculate days since start (assuming start date is January 1, 2024)
    start_date = datetime.datetime(2024, 1, 1)
    days_since_start = (current_datetime - start_date).days
    
    # Get sender and receiver info
    sender_info = f"{form_data.get('senderBankLocation')} - {form_data.get('currency')}"
    receiver_info = f"{form_data.get('receiverBankLocation')} - {form_data.get('currency')}"
    
    # Transform sender and receiver info using label encoders
    try:
        sender_encoded = le_sender.transform([sender_info])[0]
        receiver_encoded = le_receiver.transform([receiver_info])[0]
    except Exception as e:
        print(f"Error encoding sender/receiver info: {str(e)}")
        sender_encoded = 0
        receiver_encoded = 0
    
    # Initialize payment type flags
    payment_types = {
        'Payment_type_Cash Withdrawal': False,
        'Payment_type_Cheque': False,
        'Payment_type_Credit card': False,
        'Payment_type_Cross-border': False,
        'Payment_type_Debit card': False
    }
    
    # Set the selected payment type to true
    selected_payment_type = form_data.get('paymentType')
    if selected_payment_type:
        payment_type_key = f'Payment_type_{selected_payment_type}'
        if payment_type_key in payment_types:
            payment_types[payment_type_key] = True
    
    # Create the final transaction data
    transaction_data = {
        'Time': time_in_seconds,
        'Sender_account': form_data.get('senderAccount'),
        'Receiver_account': form_data.get('receiverAccount'),
        'Amount': float(form_data.get('amount', 0)),
        'Days_Since_Start': days_since_start,
        'Year': formatted_date['year'],
        'Month': formatted_date['month'],
        'Day': formatted_date['day'],
        **payment_types,
        'Sender_Info': sender_info,
        'Receiver_Info': receiver_info,
        'Sender_Info_Encoded': int(sender_encoded),
        'Receiver_Info_Encoded': int(receiver_encoded)
    }
    
    return transaction_data

def format_transaction_for_display(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format transaction data for display in the frontend.
    
    Args:
        transaction_data (Dict[str, Any]): Preprocessed transaction data
        
    Returns:
        Dict[str, Any]: Formatted transaction data for display
    """
    # Create a copy of the transaction data
    display_data = transaction_data.copy()
    
    # Format amount with currency symbol
    display_data['Amount'] = f"${display_data['Amount']:,.2f}"
    
    # Format date
    display_data['Date'] = f"{display_data['Year']}-{display_data['Month']:02d}-{display_data['Day']:02d}"
    
    # Add encoded info section
    display_data['Encoded_Info'] = {
        'Original Sender Info': display_data['Sender_Info'],
        'Encoded Sender Info': display_data['Sender_Info_Encoded'],
        'Original Receiver Info': display_data['Receiver_Info'],
        'Encoded Receiver Info': display_data['Receiver_Info_Encoded']
    }
    
    return display_data 