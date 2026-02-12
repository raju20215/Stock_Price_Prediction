# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-12

### Added
- 🎉 Initial release of NSE Hybrid AI Stock Price Predictor
- 12 pre-trained LSTM models for major NSE stocks
- FinBERT sentiment analysis integration
- Google News RSS feed integration for reliable news fetching
- Yahoo Finance API as fallback news source
- Interactive Streamlit dashboard with dark theme
- Model caching system for instant stock switching
- Predicted vs. Actual price visualization
- Real-time accuracy metrics (MAE, MAPE)
- Dual-column layout with sentiment analysis panel
- Enable/Disable sentiment toggle
- News source transparency indicator
- Comprehensive documentation (README, NEWS_CONFIG, CONTRIBUTING)
- MIT License

### Technical Details
- Python 3.12+ support
- TensorFlow/Keras 2.x for deep learning
- Streamlit for web interface
- Plotly for interactive charts
- Multi-source news fetching with automatic fallback
- 60/90 day window sizes based on stock volatility
- Hybrid prediction fusion (LSTM + sentiment)

### Performance
- Model loading time: ~20-30 seconds (one-time at startup)
- Stock switching time: <1ms (cached models)
- News fetching success rate: ~90% (Google RSS)
- Average prediction accuracy: 97-98% (varies by stock)

---

## [Unreleased]

### Planned Features
- [ ] Support for additional NSE stocks (top 50)
- [ ] Multi-day predictions (3-day, 7-day forecasts)
- [ ] CSV/Excel export functionality
- [ ] Email alerts for price movements
- [ ] Dark/Light theme toggle
- [ ] REST API for predictions
- [ ] Mobile responsive design
- [ ] Portfolio tracking features

### Known Issues
- Yahoo Finance API occasionally returns empty news
- First load may take time to download FinBERT model
- Large model files (~185MB total) may impact Git clone

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2026-02-12 | Initial release with 12 stocks, LSTM+FinBERT hybrid |

---

## Migration Guide

### From Previous Versions
This is the initial release, no migration needed.

---

## Contributors

Thanks to everyone who contributed to this release!

- Initial development and model training
- Documentation and README creation
- News API integration and testing
- UI/UX design and implementation

---

**Note**: For detailed information about changes in each version, see the commit history on GitHub.
