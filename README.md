# FraudGuard - AI-Powered Fraud Detection System

A comprehensive fraud detection system that uses machine learning and AI to analyze and detect fraudulent transactions in real-time.

## Features

- Real-time transaction analysis
- Support for multiple transaction types (Credit Card, Bank Transfer)
- Integration with Google's Gemini AI for advanced fraud detection
- User-friendly dashboard interface
- Admin panel for transaction monitoring
- Detailed transaction analysis and reporting

## Tech Stack

- Python 3.x
- Flask (Web Framework)
- Google Generative AI (Gemini)
- Machine Learning Models (Random Forest)
- HTML/CSS/JavaScript (Frontend)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fraudguard.git
cd fraudguard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Create a `.env` file in the root directory
- Add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Log in with the following credentials:
- Username: admin
- Password: admin123

## Project Structure

```
fraudguard/
├── app.py                 # Main Flask application
├── preprocessing.py       # Data preprocessing utilities
├── templates/            # HTML templates
├── static/              # Static files (CSS, JS, images)
├── models/              # Trained ML models
└── datasets/            # Dataset files
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 