# Contributing to NSE Hybrid AI Stock Price Predictor

First off, thank you for considering contributing to this project! 🎉

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Guidelines](#coding-guidelines)
- [Submitting Changes](#submitting-changes)

## 📜 Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## 🤝 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected behavior** vs. **actual behavior**
- **Screenshots** if applicable
- **Environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- **Clear use case** for the enhancement
- **Why this would be useful** to most users
- **Possible implementation** ideas (optional)

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code, add tests if applicable
3. Ensure your code follows the style guidelines
4. Update documentation as needed
5. Issue the pull request!

## 🛠️ Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/nse-stock-predictor.git
cd nse-stock-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📝 Coding Guidelines

### Python Style

- Follow **PEP 8** guidelines
- Use **type hints** where possible
- Add **docstrings** to all functions
- Keep functions **focused and small**
- Use **meaningful variable names**

### Example:

```python
def calculate_sentiment(headlines: list[str], model) -> float:
    """
    Calculate sentiment score from news headlines using FinBERT.
    
    Args:
        headlines: List of news headline strings
        model: Pre-loaded FinBERT model
    
    Returns:
        float: Sentiment score between -1.0 and 1.0
    """
    # Implementation here
    pass
```

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Be descriptive but concise
- Reference issues when applicable

Examples:
```
✅ Add support for BSE stocks
✅ Fix sentiment calculation for empty headlines
✅ Update README with installation instructions
✅ Refactor model loading for better performance
```

## 🚀 Submitting Changes

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit:
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** on GitHub

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Self-reviewed the code
- [ ] Commented complex parts
- [ ] Updated documentation
- [ ] No new warnings
- [ ] Tested locally

## 🎯 Areas for Contribution

### High Priority

- [ ] Add more NSE stocks
- [ ] Improve sentiment analysis accuracy
- [ ] Add unit tests
- [ ] Performance optimization

### Medium Priority

- [ ] Multi-day predictions
- [ ] Export functionality
- [ ] Mobile responsive design
- [ ] Additional visualizations

### Low Priority

- [ ] Dark/Light theme
- [ ] Email notifications
- [ ] Portfolio tracking
- [ ] API development

## 💡 Questions?

Feel free to open an issue with the `question` label or reach out to the maintainers.

Thank you for contributing! 🙏
